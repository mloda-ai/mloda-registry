"""Fixtures shared by the OtelExtender and OtelMetricsExtender test modules."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mloda.testing.extenders.otel import make_span_capture


@pytest.fixture
def otel_capture() -> Iterator[tuple[TracerProvider, InMemorySpanExporter]]:
    """A fresh, isolated (provider, exporter) pair per test; never touches the global provider."""
    provider, exporter = make_span_capture()
    yield provider, exporter
    provider.shutdown()
