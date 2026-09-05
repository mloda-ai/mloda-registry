"""Self-tests for mloda.testing.extenders.otel helpers."""

from __future__ import annotations

import re

import pytest

pytest.importorskip("opentelemetry.sdk")

from mloda.testing.extenders.contract import ExtenderContractTestMixin
from mloda.testing.extenders.otel import (
    OtelExtenderTestMixin,
    inject_parent_carrier,
    make_span_capture,
    single_span,
    single_span_attributes,
)

_TRACEPARENT_PATTERN = re.compile(r"^00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$")


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

    @pytest.mark.parametrize(
        "name",
        [
            "test_otel_one_span_per_call",
            "test_otel_span_names_per_hook",
            "test_otel_wrapped_failure_marks_span_error_and_propagates",
            "test_otel_wrapped_failure_logs_warning_naming_extender",
            "test_otel_exception_message_never_leaks_into_span",
            "test_otel_carrier_parents_span",
            "test_otel_empty_carrier_falls_through",
            "test_otel_run_id_derives_trace_id_without_carrier",
            "test_otel_carrier_wins_over_run_id",
            "test_otel_run_all_spans_share_one_trace_id",
        ],
    )
    def test_otel_test_methods_exist(self, name: str) -> None:
        assert hasattr(OtelExtenderTestMixin, name)
