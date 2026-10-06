"""OtelLogAuditSink: an AuditSink that emits each audit record as an OpenTelemetry log record."""

from __future__ import annotations

import calendar
import hashlib
import hmac
import json
import logging
import threading
from collections.abc import Mapping
from typing import Any

from mloda.community.extenders.shared.teardown import (
    CLOSE_TIMEOUT,
    capped_close_timeout,
    force_flush,
    to_timeout_millis,
)
from mloda.enterprise.extenders.audit._records import _is_blank, _parse_event_time
from mloda.enterprise.extenders.audit._signers import _MIN_KEY_BYTES

logger = logging.getLogger(__name__)

_LOGGER_NAME = "mloda_enterprise_audit"

# Audit record key to log attribute name; only these keys are ever forwarded.
_STR_ATTRIBUTES = {
    "decision": "mloda.audit.decision",
    "deny_reason": "mloda.audit.deny_reason",
    "policy_version": "mloda.audit.policy_version",
    "hook": "mloda.audit.hook",
    "run_id": "mloda.run.id",
    "plan_id": "mloda.plan.id",
    "structure_hash": "mloda.plan.structure_hash",
    "tenant_id": "mloda.tenant.id",
    "project_id": "mloda.project.id",
    "feature_group_class": "mloda.feature_group.name",
    "error_type": "error.type",
    "phase": "mloda.audit.phase",
    "step_run_id": "mloda.step.run_id",
}
_ENFORCED_ATTRIBUTE = "mloda.audit.enforced"
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
    # Integer arithmetic: a float timestamp loses microsecond precision.
    parsed = _parse_event_time(event_time)
    return calendar.timegm(parsed.utctimetuple()) * 10**9 + parsed.microsecond * 1000


def _attributes(record: Mapping[str, Any], key: bytes | None) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    for record_key, name in _STR_ATTRIBUTES.items():
        value = record.get(record_key)
        if not _is_blank(value):
            attributes[name] = value
    enforced = record.get("enforced")
    if isinstance(enforced, bool):
        attributes[_ENFORCED_ATTRIBUTE] = enforced
    principal: Any = record.get("principal")
    if not _is_blank(principal):
        if key is not None:
            tenant = record.get("tenant_id")
            data = json.dumps(
                [None if _is_blank(tenant) else tenant, principal], separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
            attributes[_PRINCIPAL_ATTRIBUTE] = hmac.new(key, data, hashlib.sha256).hexdigest()
    feature_names = record.get("feature_names")
    if feature_names:
        attributes[_FEATURE_NAMES_ATTRIBUTE] = list(feature_names)
    return attributes


def _is_hex(value: str, length: int) -> bool:
    return len(value) == length and all(c in "0123456789abcdefABCDEF" for c in value) and int(value, 16) != 0


def _correlation_context(record: Mapping[str, Any]) -> Any:
    trace_id: str | None = record.get("trace_id")
    span_id: str | None = record.get("span_id")
    if trace_id is None or span_id is None or _is_blank(trace_id) or _is_blank(span_id):
        return None
    if not (_is_hex(trace_id, 32) and _is_hex(span_id, 16)):
        return None
    from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags, set_span_in_context

    span_context = SpanContext(
        trace_id=int(trace_id, 16),
        span_id=int(span_id, 16),
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    return set_span_in_context(NonRecordingSpan(span_context))


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
    """Emits one OTel log record per audit record, best effort. The principal is exported only as user.hash,
    an HMAC-SHA256 keyed by user_hash_key over [tenant_id, principal]; without a key user.hash is omitted."""

    close_timeout: float = CLOSE_TIMEOUT

    def __init__(self, user_hash_key: bytes | None = None) -> None:
        try:
            import opentelemetry._logs  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "OtelLogAuditSink needs the 'opentelemetry-api' package: pip install mloda-enterprise[otel]"
            ) from exc
        if user_hash_key is not None and (not isinstance(user_hash_key, bytes) or len(user_hash_key) < _MIN_KEY_BYTES):
            raise ValueError(f"OtelLogAuditSink user_hash_key must be bytes of at least {_MIN_KEY_BYTES} bytes")
        self._user_hash_key = user_hash_key

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def write(self, record: Mapping[str, Any]) -> None:
        try:
            from opentelemetry._logs import LogRecord, SeverityNumber, get_logger_provider

            provider = get_logger_provider()
            _warn_once_without_sdk(provider)
            event_time = record.get("event_time")
            deny = record.get("decision") == "deny"
            context = _correlation_context(record)
            provider.get_logger(_LOGGER_NAME).emit(
                LogRecord(
                    **({} if context is None else {"context": context}),
                    timestamp=None if event_time is None else _epoch_ns(event_time),
                    severity_number=SeverityNumber.WARN if deny else SeverityNumber.INFO,
                    severity_text="WARN" if deny else "INFO",
                    body=record.get("decision"),
                    attributes=_attributes(record, self._user_hash_key),
                )
            )
        except Exception as exc:
            logger.warning("%s failed to emit an audit log record: %s", type(self).__name__, type(exc).__name__)

    def flush(self) -> None:
        """Called by AuditExtender.close() on graceful MULTIPROCESSING worker exit; flushes the resolved
        logger provider within close_timeout and the remaining close budget, best effort like write()."""
        try:
            from opentelemetry._logs import get_logger_provider

            provider = get_logger_provider()
            result = force_flush(provider, timeout_millis=to_timeout_millis(capped_close_timeout(self.close_timeout)))
            if result is False:
                logger.warning("%s did not flush all log records within its close budget", type(self).__name__)
        except Exception as exc:
            logger.warning("%s failed to flush the logger provider: %s", type(self).__name__, type(exc).__name__)
