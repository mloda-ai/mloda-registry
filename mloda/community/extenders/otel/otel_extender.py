"""OtelExtender: emits OpenTelemetry spans for mloda pipeline hooks."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Span, Status, StatusCode

from mloda.core.abstract_plugins.hook_context import HookContext
from mloda.steward import Extender, ExtenderHook

logger = logging.getLogger(__name__)

_TRACER_NAME = "mloda_community_otel"
_CONTENT_PREVIEW_MAX_LEN = 200
_TRUTHY_ENV_VALUES = {"true", "1"}

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
    """Emits one OpenTelemetry span per wrapped hook invocation, populated from the ambient HookContext."""

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
        with tracer.start_as_current_span(span_name) as span:
            if context is not None:
                _set_context_attributes(span, context)

            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                span.set_status(Status(StatusCode.ERROR))
                span.set_attribute("error.type", f"{type(exc).__module__}.{type(exc).__qualname__}")
                logger.warning("OtelExtender %s", exc)
                raise

            if context is not None and context.hook == ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE:
                if context.rows_out is not None:
                    span.set_attribute("mloda.rows.out", context.rows_out)
                if self._content_capture_enabled():
                    span.set_attribute("mloda.content.preview", self._content_preview(result))

            return result

    def _content_capture_enabled(self) -> bool:
        if self.capture_content:
            return True
        return os.environ.get("MLODA_OTEL_TRACE_CONTENT", "").strip().lower() in _TRUTHY_ENV_VALUES

    def _content_preview(self, result: Any) -> str:
        value = self.mask(result) if self.mask is not None else result
        return str(value)[:_CONTENT_PREVIEW_MAX_LEN]


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
