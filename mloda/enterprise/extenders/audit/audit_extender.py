"""AuditExtender: writes one audit record per FEATURE_GROUP_CALCULATE_FEATURE invocation.
With fail_closed=True it also refuses at FEATURE_GROUP_MATCHED, writing a deny record."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Protocol

from mloda.steward import Extender, ExtenderHook, HookContext, WarnOncePerInstance

from mloda.community.extenders.shared.data_access_identity import resolve_data_access_identity
from mloda.community.extenders.shared.open_invocations import OpenInvocationStack
from mloda.enterprise.extenders.audit._records import _append_records as _append_records
from mloda.enterprise.extenders.audit._records import _canonical_json as _canonical_json
from mloda.enterprise.extenders.audit._records import _is_blank, _utc_now
from mloda.enterprise.extenders.audit.run_manifest import (
    ManifestSigner,
    RunAlreadySealedError,
    RunNotPendingError,
    _reject_aliased_paths,
    _signer_map,
    seal_ndjson_runs,
)

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
    quarantine_damaged_lines repairs it."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, record: Mapping[str, Any]) -> None:
        _append_records(self.path, [record])


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
    """Raised by AuditExtender for a call under a run_id it already sealed or found sealed."""


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
    lists ([] for none; a load without an identity is omitted). URI query, fragment, `;` and `&` parameters
    and user information are stripped, best effort; other identities are recorded as given, so not
    credential-free, and a sealed log cannot be redacted afterwards. Records carry policy_version (the
    given value, else a fingerprint of the constructor-supplied gate, which does not track code changes).
    Keys may be added within record_version 1; an absent key means not recorded. With audit_path,
    manifest_path and signer all given (previous_signers optional), on_run_complete auto-seals the run
    that just finished. After auto-sealing a run_id, this instance (and a copy pickled after the seal) refuses any
    further calculation under it with SealedRunRefusedError before writing anything, so re-running a prepared session
    (including a retry after a failed run, since a failed run is sealed too) fails fast: prepare a new session instead.
    With raise_on_error=False (fail_closed=False), core instead logs the refusal and runs the call unaudited, as for a
    sink failure, so that re-run leaves no record for verification to find. Another AuditExtender instance is not
    refused, and its records under a sealed run_id land outside the seal. A fail_closed=True deny record written at
    plan time is a different, recoverable case: it is refused before setup, so on_run_complete never fires for it and it is never auto-sealed at all
    (not sealed-with-strays); seal it later with seal_ndjson_runs targeted at that specific run_id (found via
    verify_ndjson_log_coverage(...).unsealed_lines), not a blanket sweep, since a blanket sweep could seal a
    different run that is still live. expected_head anchoring against a deleted or truncated manifest log is
    not part of auto-sealing; call seal_ndjson_runs/verify_ndjson_log manually with expected_head for that."""

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
        paths_given = audit_path is not None or manifest_path is not None
        if signer is None:
            if paths_given:
                raise ValueError(
                    "AuditExtender audit_path and manifest_path need a signer to auto-seal; give all three or none"
                )
            if previous_signers:
                raise ValueError("AuditExtender previous_signers needs a signer, else there is nothing to seal with")
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
        self._pickle_drop_warning = WarnOncePerInstance()
        self._sealed_run_ids: set[str] = set()
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

    def on_run_complete(self, run_id: str | None) -> None:
        """Auto-seal `run_id` when audit_path/manifest_path/signer are configured; a no-op otherwise, with
        run_id=None, or when the run wrote nothing (logged as a WARNING with the audit_path, since a missing
        audit file or a run_id with no records is worth a steward's attention). Flushes the sink first, so a
        buffered record reaches the audit file before it is sealed. A pickled or copied instance has no
        signer (see __getstate__): it warns once per copy instead of raising or sealing anything.
        RunAlreadySealedError is logged at ERROR (any record written for it after that seal lies outside it,
        verification reports it); RunNotPendingError is logged at WARNING (recoverable: the run just wrote nothing
        yet). Neither is raised. A sealed or already-sealed run_id is remembered, so a later
        call under it through this instance is refused. Every other exception, e.g. ManifestVerificationError, is not
        caught here either; core logs it at ERROR and never fails the run because of it, regardless of
        raise_on_error/fail_closed."""
        if run_id is None:
            return
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
        try:
            seal_ndjson_runs(
                self._audit_path,
                self._manifest_path,
                signer=self._signer,
                previous_signers=self._previous_signers,
                run_id=run_id,
            )
            self._sealed_run_ids.add(run_id)
        except RunAlreadySealedError:
            self._sealed_run_ids.add(run_id)
            logger.error(
                "AuditExtender: run_id %r is already sealed in manifest_path %s and was not sealed again; "
                "any record written for it after that seal lies outside it (verification reports it)",
                run_id,
                self._manifest_path,
            )
        except RunNotPendingError:
            logger.warning(
                "AuditExtender: run_id %r has no audit records to seal in audit_path %s", run_id, self._audit_path
            )

    def __getstate__(self) -> dict[str, Any]:
        """Drops the signer material so a pickled copy (e.g. into a MULTIPROCESSING worker's dispatch
        payload) carries none: on_run_complete only ever runs in the parent, never in a worker copy, and
        Ed25519Signer holds non-picklable cryptography key objects besides."""
        state = dict(self.__dict__)
        state["_signer"] = None
        state["_previous_signers"] = ()
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
            self._note_load(context, args)
            return func(*args, **kwargs)

        if context.run_id in self._sealed_run_ids:
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
                    "%s failed to write an audit record: %s: %s",
                    type(self).__name__,
                    type(sink_exc).__name__,
                    sink_exc,
                )
            raise

        # Unguarded on purpose: a sink failure here must propagate (raise_on_error controls the
        # fallback, or never_fall_back under fail_closed), never be swallowed alongside a result that
        # was already computed successfully.
        # context.status is only set by core's instrument() wrapper; without it, the call still succeeded.
        record = self._build_record(context, loads, status=context.status or "success", error_type=None)
        self.sink.write(record)
        return result

    def _note_load(self, context: HookContext, args: tuple[Any, ...]) -> None:
        loads = _open_calculates.find(self)
        if loads is None:
            logger.debug("AuditExtender: INPUT_DATA_LOAD has no enclosing open calculate invocation to attach to")
            return
        identity = resolve_data_access_identity(args, context.data_access_identity)
        if identity is None:
            return
        entry = (identity, context.data_access_format)
        if entry not in loads:
            loads.append(entry)

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
