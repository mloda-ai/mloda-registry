"""Tests for otel_multiprocessing helpers."""

from __future__ import annotations

import re
import uuid
from unittest.mock import patch

import pytest
from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace_api
from opentelemetry.context import Context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mloda.community.extenders.otel.otel_multiprocessing import (
    extract_carrier,
    force_flush,
    inject_carrier,
    trace_id_from_run_id,
)

# W3C traceparent: version-traceid-spanid-flags, all lowercase hex.
_TRACEPARENT_RE = re.compile(r"^00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$")

_KNOWN_RUN_ID = "018f1e4a-7c3b-7c3b-8c3b-1234567890ab"
_OTHER_RUN_ID = "00000000-0000-0000-0000-000000000000"


def _new_provider_and_exporter() -> tuple[TracerProvider, InMemorySpanExporter]:
    """Build an isolated SDK TracerProvider + InMemorySpanExporter pair, simulating one process."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


class TestInjectCarrier:
    def test_returns_dict_with_no_active_span(self) -> None:
        carrier = inject_carrier()

        assert isinstance(carrier, dict)
        for key, value in carrier.items():
            assert isinstance(key, str)
            assert isinstance(value, str)

    def test_carrier_reflects_active_span_trace_and_span_id(self) -> None:
        provider, _exporter = _new_provider_and_exporter()
        tracer = provider.get_tracer("test-inject")

        with tracer.start_as_current_span("active-span") as span:
            span_context = span.get_span_context()
            carrier = inject_carrier()

        assert "traceparent" in carrier
        traceparent = carrier["traceparent"]
        assert _TRACEPARENT_RE.match(traceparent), traceparent

        expected_trace_id = format(span_context.trace_id, "032x")
        expected_span_id = format(span_context.span_id, "016x")
        assert expected_trace_id in traceparent
        assert expected_span_id in traceparent


class TestExtractCarrier:
    """extract_carrier() is the inverse of inject_carrier(), reconstructing a parent Context."""

    def test_round_trip_child_span_shares_trace_id_with_new_span_id(self) -> None:
        """A child span started from an extracted carrier shares the parent's trace id; separate
        TracerProvider/exporter pairs simulate propagation across a spawned worker process boundary."""
        parent_provider, _parent_exporter = _new_provider_and_exporter()
        parent_tracer = parent_provider.get_tracer("test-extract-parent")

        with parent_tracer.start_as_current_span("parent-span") as parent_span:
            parent_span_context = parent_span.get_span_context()
            carrier = inject_carrier()

        extracted = extract_carrier(carrier)

        child_provider, child_exporter = _new_provider_and_exporter()
        child_tracer = child_provider.get_tracer("test-extract-child")

        token = otel_context.attach(extracted)
        try:
            with child_tracer.start_as_current_span("child-span", context=extracted):
                pass
        finally:
            otel_context.detach(token)

        finished_spans = child_exporter.get_finished_spans()
        assert len(finished_spans) == 1
        child = finished_spans[0]

        assert child.context.trace_id == parent_span_context.trace_id
        assert child.context.span_id != parent_span_context.span_id
        assert child.parent is not None
        assert child.parent.span_id == parent_span_context.span_id

    def test_extract_empty_dict_returns_valid_context(self) -> None:
        extracted = extract_carrier({})

        assert isinstance(extracted, Context)


class TestForceFlush:
    """force_flush() duck-types on a provider's optional force_flush method."""

    def test_calls_and_returns_true_for_real_sdk_provider(self) -> None:
        provider = TracerProvider()

        with patch.object(provider, "force_flush") as mock_force_flush:
            result = force_flush(provider)

        mock_force_flush.assert_called_once()
        assert result is True

    def test_returns_false_for_object_without_force_flush(self) -> None:
        result = force_flush(object())

        assert result is False

    def test_returns_false_for_default_proxy_tracer_provider(self) -> None:
        """The API-only default ProxyTracerProvider lacks force_flush entirely."""
        proxy_provider = otel_trace_api.get_tracer_provider()
        assert not hasattr(proxy_provider, "force_flush")  # precondition this test relies on

        result = force_flush(proxy_provider)

        assert result is False


class TestTraceIdFromRunId:
    """trace_id_from_run_id() maps a UUIDv7 run id string to its 128-bit integer value."""

    def test_matches_stdlib_uuid_int_value(self) -> None:
        expected = uuid.UUID(_KNOWN_RUN_ID).int

        assert trace_id_from_run_id(_KNOWN_RUN_ID) == expected

    def test_is_deterministic_across_calls(self) -> None:
        assert trace_id_from_run_id(_KNOWN_RUN_ID) == trace_id_from_run_id(_KNOWN_RUN_ID)

    def test_different_run_ids_produce_different_trace_ids(self) -> None:
        assert trace_id_from_run_id(_KNOWN_RUN_ID) != trace_id_from_run_id(_OTHER_RUN_ID)

    def test_invalid_uuid_string_raises_value_error(self) -> None:
        """An invalid UUID string propagates uuid.UUID's natural ValueError, unswallowed."""
        with pytest.raises(ValueError):
            trace_id_from_run_id("not-a-uuid")
