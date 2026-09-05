"""OtelExtender: emits OpenTelemetry spans for mloda pipeline hooks."""

from __future__ import annotations

import logging
import os
import reprlib
from collections.abc import Callable
from typing import Any

from mloda.steward import Extender, ExtenderHook, HookContext
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import (
    NonRecordingSpan,
    Span,
    SpanContext,
    Status,
    StatusCode,
    TraceFlags,
    TracerProvider,
    set_span_in_context,
)

from mloda.community.extenders.otel.otel_multiprocessing import extract_carrier, trace_id_from_run_id

logger = logging.getLogger(__name__)

_TRACER_NAME = "mloda_community_otel"
_CONTENT_PREVIEW_MAX_LEN = 200
_TRUTHY_ENV_VALUES = {"true", "1"}

# Fixed, nonzero placeholder span id used as the parent span id when synthesizing a NonRecordingSpan
# from a run_id (no real parent span was ever created; only the deterministic trace_id matters here).
_RUN_ID_PARENT_SPAN_ID = 0x0000000000000001

# Bounded repr for content previews: only recurses into the first N elements of a container,
# so it never materializes a full repr/str of a huge result before truncation (see _content_preview).
_BOUNDED_REPR = reprlib.Repr()
_BOUNDED_REPR.maxlevel = 3
_BOUNDED_REPR.maxlist = 10
_BOUNDED_REPR.maxdict = 10
_BOUNDED_REPR.maxset = 10
_BOUNDED_REPR.maxfrozenset = 10
_BOUNDED_REPR.maxtuple = 10
_BOUNDED_REPR.maxstring = 30
_BOUNDED_REPR.maxother = 30

_SPAN_NAMES: dict[ExtenderHook, str] = {
    ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE: "mloda.calculate",
    ExtenderHook.VALIDATE_INPUT_FEATURE: "mloda.validate.input",
    ExtenderHook.VALIDATE_OUTPUT_FEATURE: "mloda.validate.output",
}

_OPERATION_NAMES: dict[ExtenderHook, str] = {
    ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE: "calculate",
    ExtenderHook.VALIDATE_INPUT_FEATURE: "validate",
    ExtenderHook.VALIDATE_OUTPUT_FEATURE: "validate",
}


class OtelExtender(Extender):
    """Emits one OpenTelemetry span per wrapped hook invocation, populated from the ambient HookContext.
    An injected tracer_provider is process-local: pickled copies (worker processes under
    ParallelizationMode.MULTIPROCESSING) drop it and fall back to the global tracer provider."""

    def __init__(
        self,
        raise_on_error: bool = False,
        capture_content: bool = False,
        mask: Callable[[Any], Any] | None = None,
        tracer_provider: TracerProvider | None = None,
    ) -> None:
        self.raise_on_error = raise_on_error
        self.capture_content = capture_content
        self.mask = mask
        self._tracer_provider = tracer_provider

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_tracer_provider"] = None
        return state

    def wraps(self) -> set[ExtenderHook]:
        return {
            ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
            ExtenderHook.VALIDATE_INPUT_FEATURE,
            ExtenderHook.VALIDATE_OUTPUT_FEATURE,
        }

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        span_name = _SPAN_NAMES.get(context.hook, "mloda.unknown") if context is not None else "mloda.unknown"

        tracer = trace.get_tracer(_TRACER_NAME, tracer_provider=self._tracer_provider)
        parent_context = _parent_context(context)
        with tracer.start_as_current_span(span_name, record_exception=False, context=parent_context) as span:
            if context is not None:
                _set_context_attributes(span, context)

            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                span.set_status(Status(StatusCode.ERROR))
                span.set_attribute("error.type", f"{type(exc).__module__}.{type(exc).__qualname__}")
                logger.warning("OtelExtender %s failed: %s: %s", span_name, type(exc).__name__, exc)
                raise

            try:
                if context is not None and context.hook == ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE:
                    if context.rows_out is not None:
                        span.set_attribute("mloda.rows.out", context.rows_out)
                    if self._content_capture_enabled():
                        span.set_attribute("mloda.content.preview", self._content_preview(result))
            except Exception as exc:
                logger.warning("OtelExtender post-call instrumentation failed: %s: %s", type(exc).__name__, exc)

            return result

    def _content_capture_enabled(self) -> bool:
        if self.capture_content:
            return True
        return os.environ.get("MLODA_OTEL_TRACE_CONTENT", "").strip().lower() in _TRUTHY_ENV_VALUES

    def _content_preview(self, result: Any) -> str:
        value = self.mask(result) if self.mask is not None else result
        return _BOUNDED_REPR.repr(value)[:_CONTENT_PREVIEW_MAX_LEN]


def _parent_context(context: HookContext | None) -> Context | None:
    """Pick the parent context for the span to be started, by priority (highest first):

    1. context.carrier, if truthy: extracted into a real parent Context (propagated from another
       process via a W3C traceparent carrier).
    2. else context.run_id, if not None: a synthetic, non-recording parent Context whose trace_id is
       deterministically derived from run_id, so spans sharing a run_id correlate even when no carrier
       was ever exchanged.
    3. else None: today's existing default behavior (the ambient current context is used).
    """
    if context is None:
        return None

    if context.carrier:
        return extract_carrier(context.carrier)

    if context.run_id is not None:
        span_context = SpanContext(
            trace_id=trace_id_from_run_id(context.run_id),
            span_id=_RUN_ID_PARENT_SPAN_ID,
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
        return set_span_in_context(NonRecordingSpan(span_context))

    return None


def _set_context_attributes(span: Span, context: HookContext) -> None:
    span.set_attribute("mloda.operation.name", _OPERATION_NAMES.get(context.hook, "unknown"))
    span.set_attribute("mloda.feature_group.name", context.feature_group_class)
    span.set_attribute("mloda.feature_group.version", context.feature_group_version)
    span.set_attribute("mloda.compute_framework.name", context.compute_framework_name)

    if context.rows_in is not None:
        span.set_attribute("mloda.rows.in", context.rows_in)
    if len(context.feature_names) == 1:
        span.set_attribute("mloda.feature.name", context.feature_names[0])
    if context.plugin_version is not None:
        span.set_attribute("mloda.plugin.version", context.plugin_version)
    if context.run_id is not None:
        span.set_attribute("mloda.run.id", context.run_id)
    if context.worker_index is not None:
        span.set_attribute("mloda.subprocess.worker_index", context.worker_index)
