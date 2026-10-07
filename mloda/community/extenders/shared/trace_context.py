"""Active OpenTelemetry span ids; opentelemetry is optional, so it is imported lazily."""

from __future__ import annotations


def active_span_ids(*, recording_only: bool = False) -> tuple[str, str] | None:
    """(trace_id, span_id) as lowercase hex of the current valid span, or None. recording_only also requires the
    span to be recording."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        span_context = span.get_span_context()
        if not span_context.is_valid or (recording_only and not span.is_recording()):
            return None
        return format(span_context.trace_id, "032x"), format(span_context.span_id, "016x")
    except Exception:
        return None


def active_step_span_ids(step_run_id: str) -> tuple[str, str] | None:
    """Ids of the current recording span only if its mloda.step.run_id attribute equals step_run_id."""
    try:
        from opentelemetry import trace

        attributes = getattr(trace.get_current_span(), "attributes", None)
        if attributes is None or attributes.get("mloda.step.run_id") != step_run_id:
            return None
        return active_span_ids(recording_only=True)
    except Exception:
        return None
