"""AuditExtender: writes one audit record per FEATURE_GROUP_CALCULATE_FEATURE invocation."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from mloda.steward import Extender, ExtenderHook, HookContext

logger = logging.getLogger(__name__)

_ALLOWED_IDENTITY_NAMES = ("tenant_id", "project_id", "principal")


class AuditSink(Protocol):
    """Receives one audit record per calculation."""

    def write(self, record: Mapping[str, Any]) -> None: ...


class NdjsonAuditSink:
    """Appends one JSON line per record, opening the file per write so it pickles and holds no
    buffer a terminated worker could lose; ordering across concurrent writers is not guaranteed."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, record: Mapping[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


class AuditExtender(Extender):
    """Records tenant-scoped audit metadata (never values or exception messages) for every
    calculation. A missing required identity yields a deny record while the calculation still
    runs; raise_on_error=True (default) means a sink failure fails the run."""

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
        record = self._build_record(context, status=context.status, error_type=None)
        self.sink.write(record)
        return result

    def _build_record(self, context: HookContext, *, status: str | None, error_type: str | None) -> dict[str, Any]:
        missing = [name for name in self.required_identity if getattr(context, name) is None]
        return {
            "record_version": 1,
            # Audit records require the explicit Z, unlike OpenLineage's +00:00 offset.
            "event_time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
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
