"""Self-tests for mloda.testing.extenders.otel helpers, plus a minimal probe extender that
exercises the full OtelExtenderTestMixin contract independently of the real registry extender."""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

import pytest

pytest.importorskip("opentelemetry.sdk")

from mloda.steward import Extender, ExtenderHook, HookContext
from opentelemetry import propagate, trace
from opentelemetry.context import Context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import NonRecordingSpan, SpanContext, Status, StatusCode, TraceFlags, set_span_in_context

from mloda.testing.extenders.contract import ExtenderContractTestMixin
from mloda.testing.extenders.otel import (
    OtelExtenderTestMixin,
    inject_parent_carrier,
    make_span_capture,
    single_span,
    single_span_attributes,
)

_TRACEPARENT_PATTERN = re.compile(r"^00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$")

logger = logging.getLogger(__name__)

_SPAN_NAME = "probe.calculate"
# Fixed, nonzero placeholder span id: only the derived trace_id matters for a synthesized parent.
_PARENT_SPAN_ID = 0x0000000000000001


def _parent_context(context: HookContext | None) -> Context | None:
    """Carrier wins when truthy; else a non-recording span whose trace id derives from run_id; else None."""
    if context is None:
        return None
    if context.carrier:
        return propagate.extract(context.carrier)
    if context.run_id is not None:
        span_context = SpanContext(
            trace_id=uuid.UUID(context.run_id).int,
            span_id=_PARENT_SPAN_ID,
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
        return set_span_in_context(NonRecordingSpan(span_context))
    return None


class _ProbeOtelExtender(Extender):
    """Minimal OTel probe: one span per call, parented from carrier/run_id, error status on failure.
    Sink resolution: an injected tracer_provider wins, else use_sdk_defaults delegates to the ambient
    global provider, else the probe is inert and func still runs untouched."""

    def __init__(
        self,
        tracer_provider: TracerProvider | None = None,
        raise_on_error: bool = False,
        use_sdk_defaults: bool = False,
    ) -> None:
        self.raise_on_error = raise_on_error
        self.use_sdk_defaults = use_sdk_defaults
        self._tracer_provider = tracer_provider
        self._logged_inert = False

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_tracer_provider"] = None
        return state

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        if self._tracer_provider is None and not self.use_sdk_defaults:
            if not self._logged_inert:
                logger.info("_ProbeOtelExtender is inert: no injected tracer_provider and use_sdk_defaults is False")
                self._logged_inert = True
            return func(*args, **kwargs)

        context = HookContext.current()
        parent = _parent_context(context)
        tracer = trace.get_tracer("mloda-testing-probe-otel", tracer_provider=self._tracer_provider)
        with tracer.start_as_current_span(
            _SPAN_NAME, record_exception=False, context=parent, set_status_on_exception=False
        ) as span:
            try:
                return func(*args, **kwargs)
            except BaseException as exc:
                span.set_status(Status(StatusCode.ERROR))
                span.set_attribute("error.type", f"{type(exc).__module__}.{type(exc).__qualname__}")
                logger.warning("_ProbeOtelExtender %s failed: %s: %s", _SPAN_NAME, type(exc).__name__, exc)
                raise


class TestMakeSpanCapture:
    def test_finished_span_lands_in_returned_exporter(self) -> None:
        provider, exporter = make_span_capture()
        tracer = provider.get_tracer("test-extenders-otel")

        with tracer.start_as_current_span("probe-span"):
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "probe-span"


class TestSingleSpan:
    def test_raises_on_zero_spans(self) -> None:
        _, exporter = make_span_capture()
        with pytest.raises(AssertionError):
            single_span(exporter)

    def test_returns_the_span_when_exactly_one_exists(self) -> None:
        provider, exporter = make_span_capture()
        tracer = provider.get_tracer("test-extenders-otel")

        with tracer.start_as_current_span("only-span"):
            pass

        span = single_span(exporter)
        assert span.name == "only-span"


class TestSingleSpanAttributes:
    def test_returns_the_attributes_mapping(self) -> None:
        provider, exporter = make_span_capture()
        tracer = provider.get_tracer("test-extenders-otel")

        with tracer.start_as_current_span("attributed-span") as span:
            span.set_attribute("probe.key", "probe-value")

        attributes = single_span_attributes(exporter)
        assert attributes["probe.key"] == "probe-value"


class TestInjectParentCarrier:
    def test_carrier_traceparent_matches_returned_trace_and_span_ids(self) -> None:
        carrier, trace_id, span_id = inject_parent_carrier()

        traceparent = carrier["traceparent"]
        assert _TRACEPARENT_PATTERN.match(traceparent)
        _, hex_trace_id, hex_span_id, _ = traceparent.split("-")
        assert hex_trace_id == format(trace_id, "032x")
        assert hex_span_id == format(span_id, "016x")


class TestOtelExtenderTestMixinShape:
    def test_is_extender_contract_subclass(self) -> None:
        assert issubclass(OtelExtenderTestMixin, ExtenderContractTestMixin)

    def test_raise_on_error_default_is_false(self) -> None:
        assert OtelExtenderTestMixin.raise_on_error_default() is False

    def test_expected_span_names_defaults_to_none(self) -> None:
        assert OtelExtenderTestMixin.expected_span_names() is None


class TestProbeOtelExtenderContract(OtelExtenderTestMixin):
    """Self-test: _ProbeOtelExtender must satisfy every OTel contract test the mixin defines."""

    @classmethod
    def extender_class(cls) -> type[Extender]:
        return _ProbeOtelExtender

    def make_otel_extender(self, tracer_provider: TracerProvider, *, raise_on_error: bool | None = None) -> Extender:
        if raise_on_error is None:
            return _ProbeOtelExtender(tracer_provider=tracer_provider)
        return _ProbeOtelExtender(tracer_provider=tracer_provider, raise_on_error=raise_on_error)

    @classmethod
    def expected_hooks(cls) -> set[ExtenderHook] | None:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    @classmethod
    def expected_span_names(cls) -> dict[ExtenderHook, str] | None:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE: _SPAN_NAME}


class _CachedTracerProbeOtelExtender(Extender):
    """Copy of _ProbeOtelExtender that resolves its tracer once in __init__ and never calls get_tracer again."""

    def __init__(self, tracer_provider: TracerProvider | None = None, raise_on_error: bool = False) -> None:
        self.raise_on_error = raise_on_error
        self._tracer_provider = tracer_provider
        self._tracer = trace.get_tracer("mloda-testing-probe-otel-cached", tracer_provider=tracer_provider)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_tracer_provider"] = None
        state["_tracer"] = None
        return state

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        context = HookContext.current()
        parent = _parent_context(context)
        with self._tracer.start_as_current_span(
            _SPAN_NAME, record_exception=False, context=parent, set_status_on_exception=False
        ) as span:
            try:
                return func(*args, **kwargs)
            except BaseException as exc:
                span.set_status(Status(StatusCode.ERROR))
                span.set_attribute("error.type", f"{type(exc).__module__}.{type(exc).__qualname__}")
                raise


class TestOwnFailureDefaultDetectsNoFault:
    """Proves the chained own_failure() test is no longer vacuous for a probe with an already-cached tracer."""

    def test_default_own_failure_fails_loudly_when_nothing_is_faulted(self, caplog: pytest.LogCaptureFixture) -> None:
        class _Host(OtelExtenderTestMixin):
            @classmethod
            def extender_class(cls) -> type[Extender]:
                return _CachedTracerProbeOtelExtender

            def make_otel_extender(
                self, tracer_provider: TracerProvider, *, raise_on_error: bool | None = None
            ) -> Extender:
                if raise_on_error is None:
                    return _CachedTracerProbeOtelExtender(tracer_provider=tracer_provider)
                return _CachedTracerProbeOtelExtender(tracer_provider=tracer_provider, raise_on_error=raise_on_error)

        with pytest.raises(AssertionError, match="own_failure"):
            _Host().test_contract_own_failure_does_not_stop_chained_extender(caplog)
