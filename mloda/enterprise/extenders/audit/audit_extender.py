"""AuditExtender: writes one audit record per FEATURE_GROUP_CALCULATE_FEATURE invocation."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from mloda.steward import Extender, ExtenderHook, HookContext

logger = logging.getLogger(__name__)

_ALLOWED_IDENTITY_NAMES = ("tenant_id", "project_id", "principal")


class AuditSink(Protocol):
    """Receives one audit record per calculation."""

    def write(self, record: Mapping[str, Any]) -> None: ...


def _is_missing(value: str | None) -> bool:
    """A required identity value is missing when it is None or blank."""
    return value is None or not value.strip()


def _canonical_json(record: Mapping[str, Any]) -> bytes:
    """The bytes NdjsonAuditSink writes for a record, without the newline."""
    return json.dumps(record, sort_keys=True).encode("utf-8")


def _utc_now() -> str:
    # Audit records require the explicit Z, unlike OpenLineage's +00:00 offset.
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _append_records(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Append one canonical line per record through one O_APPEND descriptor."""
    data = b"".join(_canonical_json(record) + b"\n" for record in records)
    if not data:
        return
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError(f"os.write wrote no bytes to {path}")
            view = view[written:]
    finally:
        os.close(fd)


class NdjsonAuditSink:
    """One os.write per record to an O_APPEND descriptor keeps concurrent writers from interleaving
    a line, and the file is created owner-only. Opens per write, so it pickles and holds no buffer a
    terminated worker could lose; ordering across writers is not guaranteed. A short write may interleave."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, record: Mapping[str, Any]) -> None:
        _append_records(self.path, [record])


class AuditExtender(Extender):
    """Records tenant-scoped audit metadata (never values or exception messages) for every
    calculation. A missing required identity yields a deny record while the calculation still
    runs. With raise_on_error=True (default), a sink failure after a successful calculation fails
    the run; when the calculation itself fails, its exception wins and the sink failure is only logged."""

    def __init__(
        self,
        sink: AuditSink,
        required_identity: tuple[str, ...] = ("tenant_id",),
        raise_on_error: bool = True,
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
        self.sink = sink
        self.required_identity = required_identity
        self.raise_on_error = raise_on_error

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        if context is None:
            return func(*args, **kwargs)

        try:
            result = func(*args, **kwargs)
        except BaseException as exc:
            record = self._build_record(
                context, status="error", error_type=f"{type(exc).__module__}.{type(exc).__qualname__}"
            )
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

    def _build_record(self, context: HookContext, *, status: str | None, error_type: str | None) -> dict[str, Any]:
        missing = [name for name in self.required_identity if _is_missing(getattr(context, name))]
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
