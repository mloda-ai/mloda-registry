"""Helpers for propagating OTel trace context across process boundaries."""

import os
import uuid

from opentelemetry import trace as otel_trace_api
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


def env_carrier() -> dict[str, str]:
    """Carrier from TRACEPARENT/TRACESTATE (never BAGGAGE) for run_all(carrier=...); {} unless TRACEPARENT is valid."""
    carrier = {"traceparent": os.environ.get("TRACEPARENT", "")}
    if tracestate := os.environ.get("TRACESTATE"):
        carrier["tracestate"] = tracestate
    if not otel_trace_api.get_current_span(extract_carrier(carrier)).get_span_context().is_valid:
        return {}
    return carrier


def trace_id_from_run_id(run_id: str) -> int:
    """Map a UUID run id string to its 128-bit integer value."""
    return uuid.UUID(run_id).int
