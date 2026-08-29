"""Helpers for propagating OTel trace context across process boundaries."""

import uuid
from typing import Any

from opentelemetry import propagate
from opentelemetry.context import Context


def inject_carrier() -> dict[str, str]:
    """Encode the current OTel context into a W3C traceparent carrier dict."""
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    return carrier


def extract_carrier(carrier: dict[str, str]) -> Context:
    """Decode a W3C traceparent carrier dict back into an OTel Context."""
    return propagate.extract(carrier)


def force_flush(provider: Any) -> bool:
    """Call provider.force_flush() if present and callable, without raising."""
    flush = getattr(provider, "force_flush", None)
    if callable(flush):
        flush()
        return True
    return False


def trace_id_from_run_id(run_id: str) -> int:
    """Map a UUID run id string to its 128-bit integer value."""
    return uuid.UUID(run_id).int
