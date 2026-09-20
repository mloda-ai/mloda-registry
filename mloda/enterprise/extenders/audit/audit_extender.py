"""AuditExtender: writes one audit record per FEATURE_GROUP_CALCULATE_FEATURE invocation (fail_closed also
wraps FEATURE_GROUP_MATCHED)."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from mloda.steward import Extender, ExtenderHook, HookContext

from mloda.enterprise.extenders.audit._records import _append_records as _append_records
from mloda.enterprise.extenders.audit._records import _canonical_json as _canonical_json
from mloda.enterprise.extenders.audit._records import _is_blank, _utc_now

logger = logging.getLogger(__name__)

_ALLOWED_IDENTITY_NAMES = ("tenant_id", "project_id", "principal")


class AuditSink(Protocol):
    """Receives one audit record per calculation."""

    def write(self, record: Mapping[str, Any]) -> None: ...


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


class IdentityRequiredError(RuntimeError):
    """Raised by AuditExtender(fail_closed=True) when a required identity is missing."""


def _error_type(exc: BaseException) -> str:
    return f"{type(exc).__module__}.{type(exc).__qualname__}"


class AuditExtender(Extender):
    """Records tenant-scoped audit metadata (never values or exception messages) for every
    calculation. A missing required identity yields a deny record while the calculation still
    runs. With raise_on_error=True (default), a sink failure after a successful calculation fails
    the run; when the calculation itself fails, its exception wins and the sink failure is only logged.
    With fail_closed=True (needs raise_on_error=True), a missing identity writes the deny record and
    raises IdentityRequiredError before the wrapped call, also at FEATURE_GROUP_MATCHED."""

    def __init__(
        self,
        sink: AuditSink,
        required_identity: tuple[str, ...] = ("tenant_id",),
        raise_on_error: bool = True,
        fail_closed: bool = False,
    ) -> None:
        unknown = [name for name in required_identity if name not in _ALLOWED_IDENTITY_NAMES]
        if unknown:
            raise ValueError(
                f"AuditExtender required_identity has unknown name(s) {unknown}; "
                f"allowed names are {list(_ALLOWED_IDENTITY_NAMES)}"
            )
        if len(set(required_identity)) != len(required_identity):
            raise ValueError(f"AuditExtender required_identity has duplicate name(s): {required_identity}")
        if not callable(getattr(sink, "write", None)):
            raise ValueError("AuditExtender sink must implement the AuditSink protocol: a callable write(record)")
        if fail_closed and not raise_on_error:
            raise ValueError(
                "AuditExtender fail_closed=True requires raise_on_error=True: with raise_on_error=False core "
                "would log the refusal and still run the wrapped call"
            )
        if fail_closed and not required_identity:
            raise ValueError(
                "AuditExtender fail_closed=True needs a non-empty required_identity, else nothing is refused"
            )
        self.sink = sink
        self.required_identity = required_identity
        self.raise_on_error = raise_on_error
        self.fail_closed = fail_closed

    def wraps(self) -> set[ExtenderHook]:
        if self.fail_closed:
            return {ExtenderHook.FEATURE_GROUP_MATCHED, ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        if context is None:
            return func(*args, **kwargs)

        if self.fail_closed:
            missing = self._missing_identity(context)
            if missing:
                refusal = IdentityRequiredError(f"AuditExtender refused the call: missing required identity {missing}")
                # Unguarded on purpose: a sink failure must propagate instead of the refusal.
                self.sink.write(self._build_record(context, status="error", error_type=_error_type(refusal)))
                raise refusal
            if context.hook is ExtenderHook.FEATURE_GROUP_MATCHED:
                return func(*args, **kwargs)

        try:
            result = func(*args, **kwargs)
        except BaseException as exc:
            record = self._build_record(context, status="error", error_type=_error_type(exc))
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
        record = self._build_record(context, status=context.status or "success", error_type=None)
        self.sink.write(record)
        return result

    def _missing_identity(self, context: HookContext) -> list[str]:
        return [name for name in self.required_identity if _is_blank(getattr(context, name))]

    def _build_record(self, context: HookContext, *, status: str | None, error_type: str | None) -> dict[str, Any]:
        missing = self._missing_identity(context)
        return {
            "record_version": 1,
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
        }
