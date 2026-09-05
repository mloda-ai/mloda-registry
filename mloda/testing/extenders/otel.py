"""In-memory OTel span capture plus a contract mixin for extenders that emit OpenTelemetry spans."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Any
from unittest.mock import patch

import pytest
from mloda.steward import Extender, ExtenderHook
from opentelemetry import propagate
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from mloda.testing.extenders.contract import ExtenderContractTestMixin
from mloda.testing.extenders.hook_context import make_hook_context
from mloda.testing.extenders.runners import expected_value_int, run_value_int


def make_span_capture() -> tuple[TracerProvider, InMemorySpanExporter]:
    """SDK TracerProvider wired to an in-memory span exporter via a SimpleSpanProcessor."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def single_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    """Assert exactly one finished span exists and return it."""
    spans = exporter.get_finished_spans()
    assert len(spans) == 1, spans
    return spans[0]


def single_span_attributes(exporter: InMemorySpanExporter) -> Mapping[str, Any]:
    """Attributes of the single finished span."""
    attributes = single_span(exporter).attributes
    assert attributes is not None
    return attributes


def inject_parent_carrier() -> tuple[dict[str, str], int, int]:
    """Start a span on a throwaway capture provider and inject it into a W3C traceparent carrier."""
    provider, _ = make_span_capture()
    tracer = provider.get_tracer("mloda-testing-otel-carrier")
    carrier: dict[str, str] = {}
    with tracer.start_as_current_span("carrier-parent") as span:
        propagate.inject(carrier)
        span_context = span.get_span_context()
        trace_id = span_context.trace_id
        span_id = span_context.span_id
    return carrier, trace_id, span_id


class OtelExtenderTestMixin(ExtenderContractTestMixin):
    """Contract for extenders that emit OTel spans. Host provides extender_class and make_otel_extender."""

    def make_otel_extender(self, tracer_provider: TracerProvider, *, raise_on_error: bool | None = None) -> Extender:
        raise NotImplementedError

    @classmethod
    def expected_span_names(cls) -> dict[ExtenderHook, str] | None:
        """None skips the per-hook span name check; override to pin the span name for each wrapped hook."""
        return None

    @classmethod
    def raise_on_error_default(cls) -> bool:
        return False

    def make_extender(self, *, raise_on_error: bool | None = None) -> Extender:
        provider, _ = make_span_capture()
        return self.make_otel_extender(provider, raise_on_error=raise_on_error)

    def own_failure(self) -> AbstractContextManager[Any]:
        return patch.object(TracerProvider, "get_tracer", side_effect=RuntimeError("otel instrumentation boom"))

    def test_otel_one_span_per_call(self) -> None:
        provider, exporter = make_span_capture()
        extender = self.make_otel_extender(provider)

        with make_hook_context(hook=self._context_hook()).activate():
            extender(lambda: None)

        single_span(exporter)

    def test_otel_span_names_per_hook(self) -> None:
        expected = self.expected_span_names()
        if expected is None:
            pytest.skip("no expected_span_names declared")
        wraps = self.make_extender().wraps()

        for hook, name in expected.items():
            if hook not in wraps:
                continue
            provider, exporter = make_span_capture()
            extender = self.make_otel_extender(provider)
            with make_hook_context(hook=hook).activate():
                extender(lambda: None)
            assert single_span(exporter).name == name

    def test_otel_wrapped_failure_marks_span_error_and_propagates(self) -> None:
        provider, exporter = make_span_capture()
        extender = self.make_otel_extender(provider)

        def func() -> None:
            raise RuntimeError("inner boom")

        with make_hook_context(hook=self._context_hook()).activate():
            with pytest.raises(RuntimeError, match="inner boom"):
                extender(func)

        span = single_span(exporter)
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes is not None
        assert "error.type" in span.attributes

    def test_otel_wrapped_failure_logs_warning_naming_extender(self, caplog: pytest.LogCaptureFixture) -> None:
        provider, _ = make_span_capture()
        extender = self.make_otel_extender(provider)

        def func() -> None:
            raise RuntimeError("inner boom")

        with make_hook_context(hook=self._context_hook()).activate():
            with caplog.at_level(logging.WARNING):
                with pytest.raises(RuntimeError, match="inner boom"):
                    extender(func)

        extender_name = self.extender_class().__name__
        assert any(extender_name in message and "inner boom" in message for message in caplog.messages)

    def test_otel_exception_message_never_leaks_into_span(self) -> None:
        provider, exporter = make_span_capture()
        extender = self.make_otel_extender(provider)
        marker = "SENSITIVE_ROW_VALUE_xyz123"

        def func() -> None:
            raise ValueError(f"invalid value found: {marker}")

        with make_hook_context(hook=self._context_hook()).activate():
            with pytest.raises(ValueError):
                extender(func)

        span = single_span(exporter)
        assert span.attributes is not None
        for value in span.attributes.values():
            assert marker not in str(value)
        for event in span.events:
            if event.attributes is None:
                continue
            for value in event.attributes.values():
                assert marker not in str(value)

    def test_otel_carrier_parents_span(self) -> None:
        carrier, trace_id, span_id = inject_parent_carrier()
        provider, exporter = make_span_capture()
        extender = self.make_otel_extender(provider)

        with make_hook_context(hook=self._context_hook(), carrier=carrier).activate():
            extender(lambda: None)

        span = single_span(exporter)
        assert span.context is not None
        assert span.context.trace_id == trace_id
        assert span.parent is not None
        assert span.parent.span_id == span_id

    def test_otel_empty_carrier_falls_through(self) -> None:
        provider, exporter = make_span_capture()
        extender = self.make_otel_extender(provider)

        with make_hook_context(hook=self._context_hook(), carrier={}).activate():
            extender(lambda: None)

        single_span(exporter)

    def test_otel_run_id_derives_trace_id_without_carrier(self) -> None:
        run_id = "018f1e4a-7c3b-7c3b-8c3b-1234567890ab"
        provider, exporter = make_span_capture()
        extender = self.make_otel_extender(provider)

        with make_hook_context(hook=self._context_hook(), run_id=run_id, carrier=None).activate():
            extender(lambda: None)

        span = single_span(exporter)
        assert span.context is not None
        assert span.context.trace_id == uuid.UUID(run_id).int

    def test_otel_carrier_wins_over_run_id(self) -> None:
        run_id = "018f1e4a-7c3b-7c3b-8c3b-1234567890ab"
        carrier, carrier_trace_id, _ = inject_parent_carrier()
        assert carrier_trace_id != uuid.UUID(run_id).int

        provider, exporter = make_span_capture()
        extender = self.make_otel_extender(provider)

        with make_hook_context(hook=self._context_hook(), run_id=run_id, carrier=carrier).activate():
            extender(lambda: None)

        span = single_span(exporter)
        assert span.context is not None
        assert span.context.trace_id == carrier_trace_id

    def test_otel_run_all_spans_share_one_trace_id(self) -> None:
        provider, exporter = make_span_capture()
        assert run_value_int(self.make_otel_extender(provider)) == expected_value_int()

        spans = exporter.get_finished_spans()
        assert spans
        trace_ids = set()
        for span in spans:
            assert span.context is not None
            trace_ids.add(span.context.trace_id)
        assert len(trace_ids) == 1
