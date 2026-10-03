"""Helpers for propagating OTel trace context across process boundaries."""

import uuid

from opentelemetry.context import Context
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from mloda.community.extenders.shared.teardown import force_flush as force_flush

_PROPAGATOR = TraceContextTextMapPropagator()


def inject_carrier() -> dict[str, str]:
    """Encode the current OTel context into a W3C traceparent carrier dict (traceparent only, no baggage)."""
    carrier: dict[str, str] = {}
    _PROPAGATOR.inject(carrier)
    return carrier


def extract_carrier(carrier: dict[str, str]) -> Context:
    """Decode a W3C traceparent carrier dict back into an OTel Context (traceparent only, baggage is ignored)."""
    return _PROPAGATOR.extract(carrier)


def trace_id_from_run_id(run_id: str) -> int:
    """Map a UUID run id string to its 128-bit integer value."""
    return uuid.UUID(run_id).int
