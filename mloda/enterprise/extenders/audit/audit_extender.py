"""AuditExtender: writes one audit record per FEATURE_GROUP_CALCULATE_FEATURE invocation.
With fail_closed=True it also refuses at FEATURE_GROUP_MATCHED, writing a deny record."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from mloda.steward import Extender, ExtenderHook, HookContext

from mloda.community.extenders.shared.open_invocations import OpenInvocationStack
from mloda.enterprise.extenders.audit._records import _append_records as _append_records
from mloda.enterprise.extenders.audit._records import _canonical_json as _canonical_json
from mloda.enterprise.extenders.audit._records import _is_blank, _utc_now

logger = logging.getLogger(__name__)

_ALLOWED_IDENTITY_NAMES = ("tenant_id", "project_id", "principal")

_open_calculates: OpenInvocationStack[list[tuple[str, str | None]]] = OpenInvocationStack("audit_open_calculates")

_URI_SCHEME = re.compile(r"[A-Za-z][\w+.:-]*")
_QUERY_OR_FRAGMENT = re.compile(r"[?#]")
_PATH_PARAMS = re.compile(r"[;&]")
_AUTHORITY_LEAK = re.compile(r"[;&=\s]")


class AuditSink(Protocol):
    """Receives one audit record per calculation."""

    def write(self, record: Mapping[str, Any]) -> None: ...


def _require_sink_write(owner: str, sink: object) -> None:
    # A class object has a callable write too, so it is rejected explicitly.
    if isinstance(sink, type) or not callable(getattr(sink, "write", None)):
        raise ValueError(f"{owner} sink must implement the AuditSink protocol: a callable write(record)")


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
    """Writes each record to every sink in order; put durable sinks first, as the first failure stops the rest."""

    def __init__(self, *sinks: AuditSink) -> None:
        if not sinks:
            raise ValueError("TeeAuditSink needs at least one sink")
        for sink in sinks:
            _require_sink_write("TeeAuditSink", sink)
        self.sinks = sinks

    def write(self, record: Mapping[str, Any]) -> None:
        for sink in self.sinks:
            sink.write(record)


class IdentityRequiredError(RuntimeError):
    """Raised by AuditExtender(fail_closed=True) when a required identity is missing."""


def _error_type(exc: BaseException) -> str:
    return f"{type(exc).__module__}.{type(exc).__qualname__}"


def _gate_fingerprint(fail_closed: bool, required_identity: tuple[str, ...]) -> str:
    gate = {"fail_closed": fail_closed, "required_identity": sorted(required_identity)}
    return hashlib.sha256(json.dumps(gate, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _sanitize_data_access_identity(identity: str) -> str:
    # Core's own userinfo strip is greedy and can leave query text in the string, so the authority and path
    # are cut at leak markers too. Userinfo goes first: ; and & are valid inside it.
    scheme, separator, rest = identity.partition("://")
    if not separator or not _URI_SCHEME.fullmatch(scheme):
        return identity
    rest = _QUERY_OR_FRAGMENT.split(rest, maxsplit=1)[0].rpartition("@")[2]
    authority, slash, path = rest.partition("/")
    path = _PATH_PARAMS.split(path, maxsplit=1)[0]
    cut_authority = _AUTHORITY_LEAK.split(authority, maxsplit=1)[0]
    if cut_authority != authority:
        return f"{scheme}://{cut_authority}"
    return f"{scheme}://{authority}{slash}{path}"


class AuditExtender(Extender):
    """Records tenant-scoped audit metadata (never values or exception messages) for every
    calculation. A missing required identity yields a deny record while the calculation still
    runs. With raise_on_error=True (default), a sink failure after a successful calculation fails
    the run; when the calculation itself fails, its exception wins and the sink failure is only logged.
    With fail_closed=True (needs raise_on_error=True), a missing identity writes the deny record and
    raises IdentityRequiredError before the wrapped call, also at FEATURE_GROUP_MATCHED, and runs
    outermost (priority 0).
    Records also list the distinct data loads the call attempted, as index-aligned identity and format
    lists ([] for none; a load without an identity is omitted). URI query, fragment, `;` and `&` parameters
    and user information are stripped, best effort; other identities are recorded as given, so not
    credential-free, and a sealed log cannot be redacted afterwards. Records carry policy_version (the
    given value, else a fingerprint of the constructor-supplied gate, which does not track code changes).
    Keys may be added within record_version 1; an absent key means not recorded."""

    def __init__(
        self,
        sink: AuditSink,
        required_identity: tuple[str, ...] = ("tenant_id",),
        raise_on_error: bool = True,
        fail_closed: bool = False,
        policy_version: str | None = None,
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
        if fail_closed and not raise_on_error:
            raise ValueError(
                "AuditExtender fail_closed=True requires raise_on_error=True: with raise_on_error=False core "
                "would log the refusal and still run the wrapped call"
            )
        if fail_closed and not required_identity:
            raise ValueError(
                "AuditExtender fail_closed=True needs a non-empty required_identity, else nothing is refused"
            )
        if policy_version is not None and (not isinstance(policy_version, str) or _is_blank(policy_version)):
            raise ValueError(f"AuditExtender policy_version must be a non-blank str, got {policy_version!r}")
        self.sink = sink
        self.required_identity = required_identity
        self.raise_on_error = raise_on_error
        self.fail_closed = fail_closed
        self.policy_version = (
            policy_version if policy_version is not None else _gate_fingerprint(fail_closed, required_identity)
        )
        if fail_closed:
            # Core runs the lowest priority outermost; a lower-priority peer would otherwise run before the gate.
            self.priority = 0

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
        # fallback), never be swallowed alongside a result that was already computed successfully.
        # context.status is only set by core's instrument() wrapper; without it, the call still succeeded.
        record = self._build_record(context, loads, status=context.status or "success", error_type=None)
        self.sink.write(record)
        return result

    def _note_load(self, context: HookContext, args: tuple[Any, ...]) -> None:
        loads = _open_calculates.find(self)
        if loads is None:
            logger.debug("AuditExtender: INPUT_DATA_LOAD has no enclosing open calculate invocation to attach to")
            return
        if context.data_access_identity is None:
            return
        # Core passes the raw data_access first; using it avoids core's lossy greedy strip.
        raw = args[0] if args and isinstance(args[0], str) else context.data_access_identity
        entry = (_sanitize_data_access_identity(raw), context.data_access_format)
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
