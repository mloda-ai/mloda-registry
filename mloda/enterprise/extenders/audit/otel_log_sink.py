"""OtelLogAuditSink: an AuditSink that emits each audit record as an OpenTelemetry log record."""

from __future__ import annotations

import calendar
import hashlib
import logging
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from mloda.enterprise.extenders.audit._records import _is_blank

logger = logging.getLogger(__name__)

_LOGGER_NAME = "mloda_enterprise_audit"

# Audit record key to log attribute name; only these keys are ever forwarded.
_STR_ATTRIBUTES = {
    "decision": "mloda.audit.decision",
    "deny_reason": "mloda.audit.deny_reason",
    "policy_version": "mloda.audit.policy_version",
    "hook": "mloda.audit.hook",
    "run_id": "mloda.run.id",
    "tenant_id": "mloda.tenant.id",
    "project_id": "mloda.project.id",
    "feature_group_class": "mloda.feature_group.name",
    "error_type": "error.type",
}
_PRINCIPAL_ATTRIBUTE = "user.hash"
_FEATURE_NAMES_ATTRIBUTE = "mloda.feature.names"

_NO_SDK_MESSAGE = (
    "OtelLogAuditSink found no OpenTelemetry SDK logger provider; audit log records are not exported. "
    "Install an SDK LoggerProvider with an exporter via opentelemetry._logs.set_logger_provider "
    "(under MULTIPROCESSING, in each worker via child_bootstrap)."
)

_no_sdk_warned = False
_no_sdk_lock = threading.Lock()


def _epoch_ns(event_time: str) -> int:
    # Integer arithmetic: a float timestamp loses microsecond precision. removesuffix as 3.10 rejects a trailing Z.
    parsed = datetime.fromisoformat(event_time.removesuffix("Z")).replace(tzinfo=timezone.utc)
    return calendar.timegm(parsed.utctimetuple()) * 10**9 + parsed.microsecond * 1000


def _attributes(record: Mapping[str, Any]) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    for key, name in _STR_ATTRIBUTES.items():
        value = record.get(key)
        if not _is_blank(value):
            attributes[name] = value
    principal: Any = record.get("principal")
    if not _is_blank(principal):
        attributes[_PRINCIPAL_ATTRIBUTE] = hashlib.sha256(principal.encode("utf-8")).hexdigest()
    feature_names = record.get("feature_names")
    if feature_names:
        attributes[_FEATURE_NAMES_ATTRIBUTE] = list(feature_names)
    return attributes


def _warn_once_without_sdk(provider: object) -> None:
    global _no_sdk_warned
    # The API defaults (proxy and no-op providers) live in opentelemetry._logs; an SDK provider does not.
    if not type(provider).__module__.startswith("opentelemetry._logs"):
        return
    with _no_sdk_lock:
        if _no_sdk_warned:
            return
        _no_sdk_warned = True
    logger.warning(_NO_SDK_MESSAGE)


class OtelLogAuditSink:
    """Emits one OpenTelemetry log record per audit record, best effort, through the global logger provider.
    The principal is exported only as its sha256."""

    def __init__(self) -> None:
        try:
            import opentelemetry._logs  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "OtelLogAuditSink needs the 'opentelemetry-api' package: pip install mloda-enterprise[otel]"
            ) from exc

    def write(self, record: Mapping[str, Any]) -> None:
        try:
            from opentelemetry._logs import LogRecord, SeverityNumber, get_logger_provider

            provider = get_logger_provider()
            _warn_once_without_sdk(provider)
            event_time = record.get("event_time")
            deny = record.get("decision") == "deny"
            provider.get_logger(_LOGGER_NAME).emit(
                LogRecord(
                    timestamp=None if event_time is None else _epoch_ns(event_time),
                    severity_number=SeverityNumber.WARN if deny else SeverityNumber.INFO,
                    severity_text="WARN" if deny else "INFO",
                    body=record.get("decision"),
                    attributes=_attributes(record),
                )
            )
        except Exception as exc:
            logger.warning(
                "%s failed to emit an audit log record: %s: %s", type(self).__name__, type(exc).__name__, exc
            )
