"""AuditExtender: writes one audit record per FEATURE_GROUP_CALCULATE_FEATURE invocation.
With fail_closed=True it also refuses at FEATURE_GROUP_MATCHED, writing a deny record."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable, Iterable, Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, Protocol

from mloda.steward import Extender, ExtenderHook, HookContext, LifecycleOutcome, RunContext, WarnOncePerInstance

from mloda.community.extenders.shared.open_invocations import OpenInvocationStack
from mloda.enterprise.extenders.audit._core import (
    HeadAnchor,
    ManifestVerificationError,
    RunAlreadySealedError,
    RunNotPendingError,
    _check_line_cap,
    _check_log_id,
    _reject_aliased_paths,
)
from mloda.enterprise.extenders.audit._records import _append_records as _append_records
from mloda.enterprise.extenders.audit._records import _canonical_json as _canonical_json
from mloda.enterprise.extenders.audit._records import _is_blank, _utc_now
from mloda.enterprise.extenders.audit._segments import _genesis_older_than, _InterruptedRotationError, _rotate
from mloda.enterprise.extenders.audit._signers import ManifestSigner, _signer_map
from mloda.enterprise.extenders.audit.run_manifest import (
    _check_run_against_seal,
    seal_ndjson_runs,
)
from mloda.enterprise.extenders.audit.run_manifest import _is_run_sealed_unverified as _is_run_sealed_unverified

logger = logging.getLogger(__name__)

_ALLOWED_IDENTITY_NAMES = ("tenant_id", "project_id", "principal")

_open_calculates: OpenInvocationStack[list[tuple[str, str | None]]] = OpenInvocationStack("audit_open_calculates")


class AuditSink(Protocol):
    """Receives one audit record per calculation. May also define flush(): AuditExtender.close() calls
    it, if present, on graceful MULTIPROCESSING worker exit, so a buffering sink gets one last chance
    to drain before the worker terminates; on_run_complete also calls close() (and so flush()) from the
    parent process when sealing is configured, so a buffered record reaches the audit file before it is
    sealed."""

    def write(self, record: Mapping[str, Any]) -> None: ...


def _require_sink_write(owner: str, sink: object) -> None:
    # A class object has a callable write too, so it is rejected explicitly.
    if isinstance(sink, type) or not callable(getattr(sink, "write", None)):
        raise ValueError(f"{owner} sink must implement the AuditSink protocol: a callable write(record)")


def _require_signer_shape(owner: str, obj: object) -> None:
    # A class object has callable attributes too, so it is rejected explicitly, mirroring _require_sink_write.
    key_id = getattr(obj, "key_id", None)
    if (
        isinstance(obj, type)
        or not callable(getattr(obj, "sign", None))
        or not callable(getattr(obj, "verify", None))
        or not isinstance(key_id, str)
        or _is_blank(key_id)
    ):
        raise ValueError(
            f"{owner} must implement the ManifestSigner protocol: callable sign(payload), callable "
            f"verify(payload, signature) and a non-blank str key_id; got {obj!r}"
        )


class NdjsonAuditSink:
    """One os.write per record to an O_APPEND descriptor keeps concurrent writers from interleaving
    a line, and the file is created owner-only. Opens per write, so it pickles and holds no buffer a
    terminated worker could lose; ordering across writers is not guaranteed. A short write raises
    instead of finishing the line, leaving a torn line that blocks sealing and verification until
    quarantine_damaged_lines repairs it. Each write holds a shared flock (best effort) so a segment rotation
    cannot strand a record."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, record: Mapping[str, Any]) -> None:
        _check_line_cap([record])
        _append_records(self.path, [record], shared_lock=True)


class TeeAuditSink:
    """Writes each record to every sink in order; put durable sinks first, as write() stops at the first
    failure. flush() instead attempts every child and re-raises only the first error."""

    def __init__(self, *sinks: AuditSink) -> None:
        if not sinks:
            raise ValueError("TeeAuditSink needs at least one sink")
        for sink in sinks:
            _require_sink_write("TeeAuditSink", sink)
        self.sinks = sinks

    def write(self, record: Mapping[str, Any]) -> None:
        for sink in self.sinks:
            sink.write(record)

    def flush(self) -> None:
        """Flush every child that defines one, in order; a failing child does not stop the rest, and
        the first error raised is re-raised only after every child was attempted."""
        first_error: Exception | None = None
        for sink in self.sinks:
            flush = getattr(sink, "flush", None)
            if not callable(flush):
                continue
            try:
                flush()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


class IdentityRequiredError(RuntimeError):
    """Raised by AuditExtender(fail_closed=True) when a required identity is missing."""


class SealedRunRefusedError(RuntimeError):
    """Raised by AuditExtender for a call under a run_id already sealed in its manifest log."""


def _error_type(exc: BaseException) -> str:
    return f"{type(exc).__module__}.{type(exc).__qualname__}"


def _gate_fingerprint(fail_closed: bool, required_identity: tuple[str, ...]) -> str:
    gate = {"fail_closed": fail_closed, "required_identity": sorted(required_identity)}
    return hashlib.sha256(json.dumps(gate, sort_keys=True).encode("utf-8")).hexdigest()[:12]


class AuditExtender(Extender):
    """Records tenant-scoped audit metadata (never values or exception messages) for every
    calculation. A missing required identity yields a deny record while the calculation still
    runs. With raise_on_error=True (default), a sink failure after a successful calculation fails
    the run; when the calculation itself fails, its exception wins and the sink failure is only logged.
    With fail_closed=True, a missing identity writes the deny record and raises IdentityRequiredError
    before the wrapped call, also at FEATURE_GROUP_MATCHED, and runs outermost (priority 0). fail_closed=True
    declares core's never_fall_back, so raise_on_error has no effect on it: the refusal, and a sink
    failure on the refusal path or after a successful call, always propagate. fail_closed is read-only.
    Records also list the distinct data loads the call attempted, as index-aligned identity and format
    lists ([] for none; a load without an identity is omitted). The identity is core's data_access_identity,
    recorded as given, and a sealed log cannot be redacted afterwards. Records carry policy_version (the
    given value, else a fingerprint of the constructor-supplied gate, which does not track code changes).
    Keys may be added within record_version 1; an absent key means not recorded. With audit_path,
    manifest_path and signer all given (previous_signers optional), on_run_complete auto-seals the run
    that just finished, whatever its outcome. An auto-sealing instance, or a pickled copy of one, refuses a
    calculation with SealedRunRefusedError before writing anything when its run_id is already named in manifest_path
    or a retained archive of it (an unverified read, once per run per instance or copy, at its first calculation; a
    seal landing after that first calculation is not seen, so do not run one prepared auto-sealing session
    concurrently). Each run() of a prepared session gets a fresh run_id, so a rerun is audited and sealed as its own
    run; the refusal only guards a run_id already sealed. The read is unverified, so a writer to
    manifest_path can make runs be refused (availability only, it never hides a seal). With raise_on_error=False
    (fail_closed=False), core instead logs the refusal and runs the call unaudited, as for a sink failure. A manifest
    read failure is logged at WARNING and the run is audited without the check. An AuditExtender without sealing
    config cannot check and is never refused.
    A fail_closed=True deny record written at plan time is a different, recoverable case: it is refused
    before setup, so on_run_complete never fires for it and it is never auto-sealed at all (not sealed-with-strays).
    Records carry plan_id, and a record with no run_id is attributed to its plan_id by sealing and verification, so
    a seal_ndjson_runs sweep seals it under that plan_id (target that plan_id, not a blanket sweep, while another run may be live; find it via
    verify_ndjson_log_coverage(...).unsealed_lines). Auto-sealing uses the optional log_id and head_anchor (each new head is
    emitted to it, and its latest head must still be in the log). A seal failure (any sealing or anchor error, or a
    mismatch with an existing seal) increments the public seal_failures counter and follows seal_failure_policy:
    "log" (default), "raise", or a callable(run_id, exc). Core contains an exception raised from on_run_complete,
    so "raise" does not fail the finished run. segment_max_bytes / segment_max_age (need log_id) rotate the segment
    after an auto-seal once the sealed bytes a rotation would archive reach that size (carried pending runs do not
    count) or the segment that age; a rotation failure counts in seal_failures and follows seal_failure_policy.
    seal_index_path opts into a rebuildable seal index cache; it needs the sealing config and must not alias
    audit_path, manifest_path or the anchor path."""

    def __init__(
        self,
        sink: AuditSink,
        required_identity: tuple[str, ...] = ("tenant_id",),
        raise_on_error: bool = True,
        fail_closed: bool = False,
        policy_version: str | None = None,
        audit_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
        signer: ManifestSigner | None = None,
        previous_signers: Iterable[ManifestSigner] = (),
        log_id: str | None = None,
        head_anchor: HeadAnchor | None = None,
        seal_failure_policy: Literal["log", "raise"] | Callable[[str, BaseException], None] = "log",
        seal_index_path: str | Path | None = None,
        segment_max_bytes: int | None = None,
        segment_max_age: timedelta | None = None,
    ) -> None:
        unknown = [name for name in required_identity if name not in _ALLOWED_IDENTITY_NAMES]
        if unknown:
            raise ValueError(
                f"AuditExtender required_identity has unknown name(s) {unknown}; "
                f"allowed names are {list(_ALLOWED_IDENTITY_NAMES)}"
            )
        if len(set(required_identity)) != len(required_identity):
            raise ValueError(f"AuditExtender required_identity has duplicate name(s): {required_identity}")
        _require_sink_write("AuditExtender", sink)
        if fail_closed and not required_identity:
            raise ValueError(
                "AuditExtender fail_closed=True needs a non-empty required_identity, else nothing is refused"
            )
        if policy_version is not None and (not isinstance(policy_version, str) or _is_blank(policy_version)):
            raise ValueError(f"AuditExtender policy_version must be a non-blank str, got {policy_version!r}")
        previous_signers = tuple(previous_signers)
        if seal_failure_policy not in ("log", "raise") and (
            isinstance(seal_failure_policy, (str, type)) or not callable(seal_failure_policy)
        ):
            raise ValueError(
                f"AuditExtender seal_failure_policy must be 'log', 'raise' or a callable(run_id, exc); "
                f"got {seal_failure_policy!r}"
            )
        if head_anchor is not None and (
            isinstance(head_anchor, type)
            or not callable(getattr(head_anchor, "write", None))
            or not callable(getattr(head_anchor, "latest", None))
        ):
            raise ValueError(
                f"AuditExtender head_anchor must implement the HeadAnchor protocol: callable write(head) and "
                f"callable latest(); got {head_anchor!r}"
            )
        paths_given = audit_path is not None or manifest_path is not None
        if signer is None:
            if paths_given:
                raise ValueError(
                    "AuditExtender audit_path and manifest_path need a signer to auto-seal; give all three or none"
                )
            if previous_signers:
                raise ValueError("AuditExtender previous_signers needs a signer, else there is nothing to seal with")
            if (
                log_id is not None
                or head_anchor is not None
                or seal_failure_policy != "log"
                or seal_index_path is not None
                or segment_max_bytes is not None
                or segment_max_age is not None
            ):
                raise ValueError(
                    "AuditExtender log_id, head_anchor, seal_failure_policy, seal_index_path, segment_max_bytes and "
                    "segment_max_age need the sealing config "
                    "(audit_path, manifest_path and signer), else there is nothing to seal"
                )
        elif audit_path is None or manifest_path is None:
            raise ValueError(
                "AuditExtender signer needs both audit_path and manifest_path to auto-seal; give all three or none"
            )
        else:
            # Validated up front: _signer_map's AttributeError for a non-signer-shaped object is confusing.
            _require_signer_shape("AuditExtender signer", signer)
            for previous in previous_signers:
                _require_signer_shape("AuditExtender previous_signers entry", previous)
            # Reuse seal_ndjson_runs's own checks so a misconfiguration fails at construction, not at run end.
            _reject_aliased_paths(audit_path=audit_path, manifest_path=manifest_path)
            _signer_map(signer, previous_signers)
            _check_log_id("AuditExtender", log_id)
            if segment_max_bytes is not None or segment_max_age is not None:
                if log_id is None:
                    raise ValueError("AuditExtender segment_max_bytes and segment_max_age need a log_id to rotate")
                if segment_max_bytes is not None and (
                    isinstance(segment_max_bytes, bool)
                    or not isinstance(segment_max_bytes, int)
                    or segment_max_bytes <= 0
                ):
                    raise ValueError(f"AuditExtender segment_max_bytes must be an int > 0, got {segment_max_bytes!r}")
                if segment_max_age is not None and (
                    not isinstance(segment_max_age, timedelta) or segment_max_age <= timedelta(0)
                ):
                    raise ValueError(f"AuditExtender segment_max_age must be a timedelta > 0, got {segment_max_age!r}")
            if seal_index_path is not None:
                anchor_path = getattr(head_anchor, "_path", None)
                _reject_aliased_paths(
                    audit_path=audit_path,
                    manifest_path=manifest_path,
                    seal_index_path=seal_index_path,
                    **({"head_anchor": anchor_path} if anchor_path is not None else {}),
                )
        self.sink = sink
        self.required_identity = required_identity
        self.raise_on_error = raise_on_error
        self._fail_closed = fail_closed
        self.policy_version = (
            policy_version if policy_version is not None else _gate_fingerprint(fail_closed, required_identity)
        )
        self._audit_path = audit_path
        self._manifest_path = manifest_path
        self._signer = signer
        self._previous_signers = previous_signers
        self._log_id = log_id
        self._head_anchor = head_anchor
        self._seal_failure_policy = seal_failure_policy
        self._seal_index_path = seal_index_path
        self._segment_max_bytes = segment_max_bytes
        self._segment_max_age = segment_max_age
        self.seal_failures = 0
        self._pickle_drop_warning = WarnOncePerInstance()
        self._run_sealed: dict[str, bool] = {}
        if fail_closed:
            # Core runs the lowest priority outermost; a lower-priority peer would otherwise run before the gate.
            self.priority = 0
            self.never_fall_back = True

    @property
    def fail_closed(self) -> bool:
        """Read-only: fixed at construction, never a value set later."""
        return self._fail_closed

    # Core calls close() with no args on graceful MULTIPROCESSING worker exit and ignores the result.
    def close(self) -> None:
        """Flush the sink if it defines flush(); a no-op otherwise. An exception propagates (core logs
        it at ERROR)."""
        flush = getattr(self.sink, "flush", None)
        if callable(flush):
            flush()

    def on_run_complete(self, run: RunContext, outcome: LifecycleOutcome) -> None:
        """Auto-seal `run.run_id` (for any outcome status) when audit_path/manifest_path/signer are configured; a no-op otherwise, with
        run_id=None, or when the run wrote nothing (logged as a WARNING with the audit_path, since a missing
        audit file or a run_id with no records is worth a steward's attention). Flushes the sink first, so a
        buffered record reaches the audit file before it is sealed. A pickled or copied instance has no
        signer (see __getstate__): it warns once per copy instead of raising or sealing anything.
        RunAlreadySealedError is logged at INFO when audit_path's records of the run still match its seal (e.g. a
        refused re-run), else it is a seal failure with the reason (a record outside the seal, or a failed check);
        RunNotPendingError is logged at WARNING (recoverable: the run just wrote nothing
        yet). Neither is raised. The run's cached answer is dropped, so a later call under it re-reads the
        manifest log (and is refused there). A mismatch with an existing seal and every other exception (including
        anchor failures) is a seal failure: counted in seal_failures, then handled by seal_failure_policy. With
        "raise" it propagates; core logs it at ERROR and never fails the run because of it, regardless of
        raise_on_error/fail_closed. After a seal it made, it rotates the segment when segment_max_bytes /
        segment_max_age is passed; a rotation failure is a seal failure too (counted and handled by
        seal_failure_policy; the run stays sealed). With auto-rotation it also finishes an interrupted rotation (logged
        at WARNING) and retries the seal once."""
        run_id = run.run_id
        if run_id is None:
            return
        self._run_sealed.pop(run_id, None)  # the run is over: a later call re-reads the manifest log
        if self._signer is None:
            if self._audit_path is not None:
                self._pickle_drop_warning.warn_once(
                    lambda: logger.warning(
                        "AuditExtender: this instance is a pickled or copied copy and dropped its signer "
                        "(see __getstate__); auto-sealing for run_id %r is skipped. Call on_run_complete "
                        "only on the original instance that owns the signer.",
                        run_id,
                    )
                )
            return
        assert self._audit_path is not None and self._manifest_path is not None  # construction enforces this
        self.close()
        if not Path(self._audit_path).exists():
            logger.warning(
                "AuditExtender: audit_path %s does not exist; run_id %r wrote nothing to seal",
                self._audit_path,
                run_id,
            )
            return
        finish_error: Exception | None = None
        try:
            try:
                self._seal_run(run_id)
            except _InterruptedRotationError:
                if self._segment_max_bytes is None and self._segment_max_age is None:
                    raise
                logger.warning(
                    "AuditExtender: finishing an interrupted rotation of manifest_path %s before sealing run_id %r",
                    self._manifest_path,
                    run_id,
                )
                try:
                    self._rotate_now(min_bytes=self._segment_max_bytes, min_age=self._segment_max_age)
                except Exception as exc:
                    finish_error = exc
                else:
                    self._seal_run(run_id)
        except RunAlreadySealedError:
            try:
                _check_run_against_seal(
                    self._audit_path,
                    self._manifest_path,
                    run_id,
                    signer=self._signer,
                    previous_signers=self._previous_signers,
                )
            except Exception as exc:
                self._seal_failed(
                    run_id,
                    exc,
                    "AuditExtender: run_id %r is already sealed in manifest_path %s and was not sealed again; "
                    "its records in audit_path %s could not be confirmed to match that seal: %s",
                )
            else:
                logger.info(
                    "AuditExtender: run_id %r is already sealed in manifest_path %s and was not sealed again; "
                    "nothing has been written for it since",
                    run_id,
                    self._manifest_path,
                )
        except RunNotPendingError:
            logger.warning(
                "AuditExtender: run_id %r has no audit records to seal in audit_path %s", run_id, self._audit_path
            )
        except Exception as exc:
            self._seal_failed(run_id, exc)
        else:
            if finish_error is None:
                self._rotate_if_due(run_id)
        if finish_error is not None:
            self._seal_failed(run_id, finish_error, action="finishing an interrupted rotation before sealing")

    def _anchored_heads(self) -> list[str]:
        latest = self._head_anchor.latest() if self._head_anchor is not None else None
        return [latest] if latest is not None else []

    def _seal_run(self, run_id: str) -> None:
        assert self._audit_path is not None and self._manifest_path is not None and self._signer is not None
        seal_ndjson_runs(
            self._audit_path,
            self._manifest_path,
            signer=self._signer,
            previous_signers=self._previous_signers,
            run_id=run_id,
            sealed_late=False,
            log_id=self._log_id,
            head_anchor=self._head_anchor,
            anchored_heads=self._anchored_heads(),
            seal_index_path=self._seal_index_path,
        )

    def _rotate_now(self, *, min_bytes: int | None, min_age: timedelta | None) -> None:
        assert self._audit_path is not None and self._manifest_path is not None
        assert self._signer is not None and self._log_id is not None
        _rotate(
            self._audit_path,
            self._manifest_path,
            signer=self._signer,
            previous_signers=self._previous_signers,
            log_id=self._log_id,
            anchored_heads=self._anchored_heads(),
            head_anchor=self._head_anchor,
            min_archived_bytes=min_bytes,
            min_age=min_age,
        )

    def _rotate_if_due(self, run_id: str) -> None:
        if self._segment_max_bytes is None and self._segment_max_age is None:
            return
        assert self._audit_path is not None and self._manifest_path is not None
        try:
            due = self._segment_max_bytes is not None and os.path.getsize(self._audit_path) >= self._segment_max_bytes
            if not due and self._segment_max_age is not None:
                with open(self._manifest_path, encoding="utf-8") as handle:
                    due = _genesis_older_than(json.loads(handle.readline()), self._segment_max_age)
            if due:
                self._rotate_now(min_bytes=self._segment_max_bytes, min_age=self._segment_max_age)
        except Exception as exc:
            self._seal_failed(run_id, exc, action="rotating the segment after sealing")

    def _seal_failed(
        self, run_id: str, exc: BaseException, mismatch_message: str | None = None, *, action: str = "sealing"
    ) -> None:
        self.seal_failures += 1
        policy = self._seal_failure_policy
        if policy == "raise":
            raise exc
        if callable(policy):
            policy(run_id, exc)
            return
        if mismatch_message is not None:
            logger.error(mismatch_message, run_id, self._manifest_path, self._audit_path, exc)
        elif isinstance(exc, ManifestVerificationError):
            logger.error(
                "AuditExtender: %s run_id %r failed in manifest_path %s for audit_path %s (%s): %s",
                action,
                run_id,
                self._manifest_path,
                self._audit_path,
                type(exc).__name__,
                exc,
            )
        else:
            logger.error(
                "AuditExtender: %s run_id %r failed in manifest_path %s for audit_path %s (%s)",
                action,
                run_id,
                self._manifest_path,
                self._audit_path,
                type(exc).__name__,
            )

    def __getstate__(self) -> dict[str, Any]:
        """Drops the signer material, head anchor and failure policy so a pickled copy (e.g. into a
        MULTIPROCESSING worker's dispatch payload) carries none: on_run_complete only ever runs in the parent,
        never in a worker copy, and Ed25519Signer holds non-picklable cryptography key objects besides. Also resets the per-run
        cache, so the copy carries no run_ids and re-checks the manifest log."""
        state = dict(self.__dict__)
        state["_signer"] = None
        state["_previous_signers"] = ()
        state["_head_anchor"] = None
        state["_seal_failure_policy"] = "log"
        state["_run_sealed"] = {}
        return state

    def wraps(self) -> set[ExtenderHook]:
        if self.fail_closed:
            return {
                ExtenderHook.FEATURE_GROUP_MATCHED,
                ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
                ExtenderHook.INPUT_DATA_LOAD,
            }
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, ExtenderHook.INPUT_DATA_LOAD}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        if context is None:
            return func(*args, **kwargs)

        # A load only runs inside a calculate call that already passed the gate.
        if context.hook is ExtenderHook.INPUT_DATA_LOAD:
            self._note_load(context)
            return func(*args, **kwargs)

        # MATCHED runs at plan time under a freshly minted run_id, so it is never checked and never caches.
        if context.hook is ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE and self._found_sealed(context.run_id):
            raise SealedRunRefusedError(
                f"AuditExtender refused the call: run_id {context.run_id!r} is already sealed in the "
                "manifest log; prepare a new session to run again"
            )

        if self.fail_closed:
            missing = self._missing_identity(context)
            if missing:
                try:
                    raise IdentityRequiredError(f"AuditExtender refused the call: missing required identity {missing}")
                except IdentityRequiredError as refusal:
                    # Unguarded on purpose: a sink failure must propagate (chained to the refusal), never be swallowed.
                    self.sink.write(self._build_record(context, [], status="error", error_type=_error_type(refusal)))
                    raise
            if context.hook is ExtenderHook.FEATURE_GROUP_MATCHED:
                return func(*args, **kwargs)

        loads: list[tuple[str, str | None]] = []
        try:
            with _open_calculates.open(self, loads):
                result = func(*args, **kwargs)
        except BaseException as exc:
            record = self._build_record(context, loads, status="error", error_type=_error_type(exc))
            try:
                self.sink.write(record)
            except Exception as sink_exc:
                logger.warning(
                    "%s failed to write an audit record: %s",
                    type(self).__name__,
                    type(sink_exc).__name__,
                )
            raise

        # Unguarded on purpose: a sink failure here must propagate (raise_on_error controls the
        # fallback, or never_fall_back under fail_closed), never be swallowed alongside a result that
        # was already computed successfully.
        # context.status is only set by core's instrument() wrapper; without it, the call still succeeded.
        record = self._build_record(context, loads, status=context.status or "success", error_type=None)
        self.sink.write(record)
        return result

    def _note_load(self, context: HookContext) -> None:
        loads = _open_calculates.find(self)
        if loads is None:
            logger.debug("AuditExtender: INPUT_DATA_LOAD has no enclosing open calculate invocation to attach to")
            return
        identity = context.data_access_identity
        if identity is None:
            return
        entry = (identity, context.data_access_format)
        if entry not in loads:
            loads.append(entry)

    def _found_sealed(self, run_id: str | None) -> bool:
        """Whether run_id is sealed in the manifest log; cached per run until on_run_complete, per
        instance or copy."""
        if self._manifest_path is None or run_id is None:
            return False
        if run_id in self._run_sealed:
            return self._run_sealed[run_id]
        try:
            found = _is_run_sealed_unverified(self._manifest_path, run_id, self._seal_index_path)
        except OSError as exc:
            logger.warning(
                "AuditExtender: could not read manifest_path %s for run_id %r (%s); calculations under it "
                "are audited without the sealed-run check",
                self._manifest_path,
                run_id,
                type(exc).__name__,
            )
            self._run_sealed[run_id] = False
            return False
        self._run_sealed[run_id] = found
        return found

    def _missing_identity(self, context: HookContext) -> list[str]:
        return [name for name in self.required_identity if _is_blank(getattr(context, name))]

    def _build_record(
        self, context: HookContext, loads: list[tuple[str, str | None]], *, status: str | None, error_type: str | None
    ) -> dict[str, Any]:
        missing = self._missing_identity(context)
        return {
            "record_version": 1,
            "policy_version": self.policy_version,
            "event_time": _utc_now(),
            "run_id": context.run_id,
            "plan_id": context.plan_id,
            "tenant_id": context.tenant_id,
            "project_id": context.project_id,
            "principal": context.principal,
            "decision": "deny" if missing else "allow",
            "compliant": not missing,
            "deny_reason": ("missing_" + "_and_".join(missing)) if missing else None,
            "hook": context.hook.name,
            "feature_group_class": context.feature_group_class,
            "feature_group_version": context.feature_group_version,
            "plugin_version": context.plugin_version,
            "feature_names": list(context.feature_names),
            "input_features": sorted(context.input_features) if context.input_features is not None else None,
            "compute_framework_name": context.compute_framework_name,
            "rows_out": context.rows_out,
            "duration_seconds": context.duration_seconds,
            "status": status,
            "error_type": error_type,
            "data_access_identity": [identity for identity, _ in loads],
            "data_access_format": [fmt for _, fmt in loads],
        }
