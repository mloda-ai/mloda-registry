"""OtelExtender: emits OpenTelemetry spans for mloda pipeline hooks."""

from __future__ import annotations

import logging
import os
import re
import reprlib
from collections.abc import Callable, Mapping
from typing import Any

from mloda.steward import Extender, ExtenderHook, HookContext, WarnOncePerInstance, pickle_failure_reason
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
from mloda.community.extenders.shared.bound_method import bound_method, class_attribute
from mloda.community.extenders.shared.data_access_identity import resolve_data_access_identity
from mloda.community.extenders.shared.teardown import CLOSE_TIMEOUT, force_flush, to_timeout_millis

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

_NOOP_TRACER_PROVIDER = trace.NoOpTracerProvider()

# ProxyTracerProvider is returned while no global provider is set; NoOpTracerProvider only if installed deliberately.
_API_DEFAULT_PROVIDER_TYPES = (trace.ProxyTracerProvider, trace.NoOpTracerProvider)

_INERT_MESSAGE = (
    "OtelExtender is inert: no injected tracer_provider and use_sdk_defaults is False; no spans will be "
    "emitted. Inject a tracer_provider, or pass use_sdk_defaults=True and configure an OpenTelemetry SDK "
    "tracer provider, to enable emission."
)

_NO_SDK_PROVIDER_MESSAGE = (
    "OtelExtender found no OpenTelemetry SDK tracer provider while use_sdk_defaults is True; no spans will "
    "be exported. Configure an SDK TracerProvider with an exporter via opentelemetry.trace.set_tracer_provider "
    "(install opentelemetry-sdk if missing; under MULTIPROCESSING, in each worker via child_bootstrap), "
    "or inject a tracer_provider."
)

_SPAN_NAMES: dict[ExtenderHook, str] = {
    ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE: "mloda.calculate",
    ExtenderHook.VALIDATE_INPUT_FEATURE: "mloda.validate.input",
    ExtenderHook.VALIDATE_OUTPUT_FEATURE: "mloda.validate.output",
    ExtenderHook.INPUT_DATA_LOAD: "mloda.load",
}

_OPERATION_NAMES: dict[ExtenderHook, str] = {
    ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE: "calculate",
    ExtenderHook.VALIDATE_INPUT_FEATURE: "validate",
    ExtenderHook.VALIDATE_OUTPUT_FEATURE: "validate",
    ExtenderHook.INPUT_DATA_LOAD: "load",
}

# Hooks whose owning class may declare span attributes via declared_attributes(), and whose
# rows.out is recorded after the call: calculate and load, never validate.
_DECLARABLE_HOOKS = {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE, ExtenderHook.INPUT_DATA_LOAD}

# Declared attribute values kept; other types are dropped silently.
_SCALAR_TYPES = (str, bool, int, float)

# Caps declared keys so the SDK's attribute limit can't evict core attributes.
_MAX_DECLARED_KEYS = 32

# Flags a resolved identity as not URI-shaped (keyword DSN, object repr): never recorded.
_UNSAFE_IDENTITY_CHARS = re.compile(r"[=;\s]")


class OtelExtender(Extender):
    """Emits one OpenTelemetry span per wrapped hook invocation, populated from the ambient HookContext.
    Sink resolution: injected tracer_provider wins, else use_sdk_defaults (warns once if no SDK provider is set),
    else inert (no-op span).
    An injected tracer_provider that can't survive pickling (e.g. the real SDK TracerProvider, which
    holds locks) is dropped by a trial-pickle probe when a copy is made (worker processes under
    ParallelizationMode.MULTIPROCESSING), falling back to the resolution rule above; a picklable
    custom provider is kept as-is. close() flushes the resolved provider, capped at close_timeout
    (default 1s), and never calls shutdown() (core, not the extender, owns provider lifetime)."""

    close_timeout: float = CLOSE_TIMEOUT

    def __init__(
        self,
        raise_on_error: bool = False,
        capture_content: bool = False,
        mask: Callable[[Any], Any] | None = None,
        tracer_provider: TracerProvider | None = None,
        use_sdk_defaults: bool = False,
    ) -> None:
        self.raise_on_error = raise_on_error
        self.capture_content = capture_content
        self.mask = mask
        self._tracer_provider = tracer_provider
        self.use_sdk_defaults = use_sdk_defaults
        self._inert_warning = WarnOncePerInstance()  # shared by the inert and no-SDK warnings
        self._pickle_drop_warning = WarnOncePerInstance()

    def _configured_tracer_provider(self) -> TracerProvider | None:
        """Injected provider wins, else the global SDK provider when use_sdk_defaults, else None."""
        if self._tracer_provider is not None:
            return self._tracer_provider
        if self.use_sdk_defaults:
            return trace.get_tracer_provider()
        return None

    def _resolve_tracer_provider(self) -> TracerProvider:
        provider = self._configured_tracer_provider()
        if provider is None:
            self._warn_once(_INERT_MESSAGE)
            return _NOOP_TRACER_PROVIDER
        if self.use_sdk_defaults and isinstance(provider, _API_DEFAULT_PROVIDER_TYPES):
            self._warn_once(_NO_SDK_PROVIDER_MESSAGE)
        return provider

    def _warn_once(self, message: str) -> None:
        self._inert_warning.warn_once(lambda: logger.warning(message))

    # Core calls close() with no args on graceful MULTIPROCESSING worker exit and ignores the result.
    def close(self) -> None:
        """Flush the resolved tracer_provider within close_timeout, best effort; never raises and never
        calls shutdown() (core, not the extender, owns provider lifetime). Inert (no injected provider,
        use_sdk_defaults False) touches no provider."""
        provider = self._configured_tracer_provider()
        if provider is None:
            return
        try:
            result = force_flush(provider, timeout_millis=to_timeout_millis(self.close_timeout))
        except Exception as exc:
            logger.warning("%s failed to flush tracer_provider: %s: %s", type(self).__name__, type(exc).__name__, exc)
            return
        if result is False:
            logger.warning("%s did not flush all spans within close_timeout", type(self).__name__)

    def __getstate__(self) -> dict[str, Any]:
        provider = self._tracer_provider
        failure_reason = pickle_failure_reason(provider) if provider is not None else None
        if failure_reason is not None:
            self._pickle_drop_warning.warn_once(
                lambda: logger.warning(
                    "OtelExtender drops an injected tracer_provider when pickled or copied because it "
                    f"isn't picklable ({failure_reason}); the copy is inert unless use_sdk_defaults=True, "
                    "which lets it resolve a provider installed in its own process, e.g. via "
                    "child_bootstrap under MULTIPROCESSING."
                )
            )
        state = dict(self.__dict__)
        if failure_reason is not None:
            state["_tracer_provider"] = None
        return state

    def wraps(self) -> set[ExtenderHook]:
        return {
            ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
            ExtenderHook.VALIDATE_INPUT_FEATURE,
            ExtenderHook.VALIDATE_OUTPUT_FEATURE,
            ExtenderHook.INPUT_DATA_LOAD,
        }

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        span_name = _SPAN_NAMES.get(context.hook, "mloda.unknown") if context is not None else "mloda.unknown"

        tracer_provider = self._resolve_tracer_provider()
        tracer = trace.get_tracer(_TRACER_NAME, tracer_provider=tracer_provider)
        parent_context = _parent_context(context)
        with tracer.start_as_current_span(
            span_name, record_exception=False, context=parent_context, set_status_on_exception=False
        ) as span:
            if context is not None:
                try:
                    _set_context_attributes(span, context)
                    if context.hook == ExtenderHook.INPUT_DATA_LOAD:
                        _set_load_attributes(span, context, args)
                except BaseException as exc:
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute("error.type", f"{type(exc).__module__}.{type(exc).__qualname__}")
                    raise
                if context.hook in _DECLARABLE_HOOKS:
                    self._set_declared_attributes(span, func, args)

            try:
                result = func(*args, **kwargs)
            except BaseException as exc:
                span.set_status(Status(StatusCode.ERROR))
                span.set_attribute("error.type", f"{type(exc).__module__}.{type(exc).__qualname__}")
                logger.warning("OtelExtender %s failed: %s: %s", span_name, type(exc).__name__, exc)
                raise

            try:
                if context is not None and context.hook in _DECLARABLE_HOOKS:
                    if context.rows_out is not None:
                        span.set_attribute("mloda.rows.out", context.rows_out)
                    if (
                        context.hook == ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE
                        and span.is_recording()
                        and self._content_capture_enabled()
                    ):
                        span.set_attribute("mloda.content.preview", self._content_preview(result))
            except Exception as exc:
                logger.warning("OtelExtender post-call instrumentation failed: %s: %s", type(exc).__name__, exc)

            return result

    def _set_declared_attributes(self, span: Span, func: Any, args: tuple[Any, ...]) -> None:
        """mloda.declared.<key> attributes from a declared_attributes classmethod on func's owning class.
        Contained like _set_context_attributes: Exception logs a WARNING and skips, interrupt marks ERROR and re-raises."""
        if not span.is_recording():
            return
        declare = class_attribute(func, "declared_attributes")
        if declare is None:
            return
        try:
            attributes = declare(Extender.feature_set(args))
            if not isinstance(attributes, Mapping):
                raise TypeError(f"declared_attributes returned {type(attributes).__name__}, expected a Mapping")
            items = list(attributes.items())
        except Exception as exc:
            owner_name = _owning_class_name(func)
            logger.warning(
                "%s declared_attributes on %s failed: %s", type(self).__name__, owner_name, type(exc).__name__
            )
            return
        except BaseException as exc:
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute("error.type", f"{type(exc).__module__}.{type(exc).__qualname__}")
            raise
        count = 0
        for key, value in items:
            if count >= _MAX_DECLARED_KEYS:
                break
            if not isinstance(value, _SCALAR_TYPES):
                continue
            if isinstance(value, str):
                value = value[:_CONTENT_PREVIEW_MAX_LEN]
            span.set_attribute(f"mloda.declared.{key}", value)
            count += 1

    def _content_capture_enabled(self) -> bool:
        if self.capture_content:
            return True
        return os.environ.get("MLODA_OTEL_TRACE_CONTENT", "").strip().lower() in _TRUTHY_ENV_VALUES

    def _content_preview(self, result: Any) -> str:
        value = self.mask(result) if self.mask is not None else result
        return _BOUNDED_REPR.repr(value)[:_CONTENT_PREVIEW_MAX_LEN]


def _parent_context(context: HookContext | None) -> Context | None:
    """Pick the parent context for the span to be started, by priority (highest first):

    1. INPUT_DATA_LOAD only: a valid ambient active span (e.g. the enclosing mloda.calculate span)
       wins and None is returned, making the load span its child. If a carrier or run_id is also set,
       the ambient span wins only when its trace id matches theirs; otherwise falls through to 2/3.
    2. context.carrier, if truthy: extracted into a real parent Context (propagated from another
       process via a W3C traceparent carrier).
    3. else context.run_id, if not None: a synthetic, non-recording parent Context whose trace_id is
       deterministically derived from run_id, so spans sharing a run_id correlate even when no carrier
       was ever exchanged.
    4. else None: today's existing default behavior (the ambient current context is used).
    """
    if context is None:
        return None

    if context.hook == ExtenderHook.INPUT_DATA_LOAD:
        ambient_span_context = trace.get_current_span().get_span_context()
        if ambient_span_context.is_valid:
            if not context.carrier and context.run_id is None:
                return None
            expected_trace_id = _expected_trace_id(context)
            if expected_trace_id is None or expected_trace_id == ambient_span_context.trace_id:
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


def _expected_trace_id(context: HookContext) -> int | None:
    """The trace id the carrier/run_id fallback rule would give (carrier wins), per _parent_context's priority."""
    if context.carrier:
        return trace.get_current_span(extract_carrier(context.carrier)).get_span_context().trace_id
    if context.run_id is not None:
        return trace_id_from_run_id(context.run_id)
    return None


def _owning_class_name(func: Any) -> str:
    """Name of the class owning func, for a warning message; falls back to func's own name."""
    owner = getattr(bound_method(func), "__self__", None)
    owning_class = owner if isinstance(owner, type) else type(owner)
    return getattr(owning_class, "__name__", repr(func))


def _set_load_attributes(span: Span, context: HookContext, args: tuple[Any, ...]) -> None:
    identity = resolve_data_access_identity(args, context.data_access_identity)
    if identity is not None and not _UNSAFE_IDENTITY_CHARS.search(identity):
        span.set_attribute("mloda.data_access.identity", identity)
    if context.data_access_format is not None:
        span.set_attribute("mloda.data_access.format", context.data_access_format)


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
