"""The supported OTel pattern under MULTIPROCESSING: a child_bootstrap installs a real provider per
spawned worker, and OtelExtender(use_sdk_defaults=True) picks it up ambiently; proven with a real
spawned worker.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("opentelemetry.sdk")

from mloda.core.runtime.flight.runner_flight_server import ParallelRunnerFlightServer
from mloda.user import ParallelizationMode
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

from mloda.community.extenders.otel import OtelExtender
from mloda.testing.extenders.otel import BATCH_SCHEDULE_DELAY_MILLIS, FileMetricExporter, FileSpanExporter
from mloda.testing.extenders.runners import expected_value_int, run_value_int


class _InstallRealProvidersBootstrap:
    """Picklable child_bootstrap (a plain callable defined at module level, not a closure): installs real
    SDK providers (a TracerProvider exporting to marker_path, optionally a MeterProvider) as the
    process-global ones inside a spawned MULTIPROCESSING worker, before that worker processes its first command.

    OtelExtender(use_sdk_defaults=True) (no injected tracer_provider) then resolves this ambiently via
    opentelemetry.trace.get_tracer_provider(), the process-global provider this bootstrap just set.

    batch=True wires a BatchSpanProcessor with a schedule_delay_millis (BATCH_SCHEDULE_DELAY_MILLIS)
    long enough to never fire on its own, so only the extender's own close() can drain the buffered
    span; shutdown_on_exit=False, because otherwise the SDK's own atexit shutdown would flush the span
    itself and hide a missing close().

    metric_marker_path additionally installs a real SDK MeterProvider as the process-global one, with a
    periodic reader whose interval never fires on its own, so only the extender's close() can export."""

    def __init__(self, marker_path: Path, batch: bool = False, metric_marker_path: Path | None = None) -> None:
        self._marker_path = marker_path
        self._batch = batch
        self._metric_marker_path = metric_marker_path

    def __call__(self) -> None:
        exporter = FileSpanExporter(self._marker_path)
        if self._batch:
            provider = TracerProvider(shutdown_on_exit=False)
            provider.add_span_processor(BatchSpanProcessor(exporter, schedule_delay_millis=BATCH_SCHEDULE_DELAY_MILLIS))
        else:
            provider = TracerProvider()
            provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        if self._metric_marker_path is not None:
            reader = PeriodicExportingMetricReader(
                FileMetricExporter(self._metric_marker_path), export_interval_millis=BATCH_SCHEDULE_DELAY_MILLIS
            )
            metrics.set_meter_provider(MeterProvider(metric_readers=[reader], shutdown_on_exit=False))


@pytest.mark.parametrize("batch", [False, True], ids=["simple", "batch"])
def test_child_bootstrap_installed_provider_emits_a_span_inside_the_spawned_worker(
    tmp_path: Path, flight_server: ParallelRunnerFlightServer, batch: bool
) -> None:
    marker_path = tmp_path / "otel_multiprocessing_spans.txt"
    metric_marker_path = tmp_path / "otel_multiprocessing_metrics.txt"
    bootstrap = _InstallRealProvidersBootstrap(marker_path, batch=batch, metric_marker_path=metric_marker_path)

    values = run_value_int(
        OtelExtender(use_sdk_defaults=True),
        parallelization_modes={ParallelizationMode.MULTIPROCESSING},
        flight_server=flight_server,
        child_bootstrap=bootstrap,
    )

    assert values == expected_value_int()
    assert marker_path.exists(), (
        "child_bootstrap's installed TracerProvider never wrote a span marker file; the spawned "
        "worker never emitted a span for OtelExtender(use_sdk_defaults=True)"
    )
    span_names = marker_path.read_text().splitlines()
    assert any(name.startswith("calculate ") for name in span_names), span_names
    assert metric_marker_path.exists(), (
        "child_bootstrap's installed MeterProvider never wrote a metric marker file; the worker's close() "
        "never flushed the meter provider for OtelExtender(use_sdk_defaults=True)"
    )
    assert "mloda.step.duration" in metric_marker_path.read_text().splitlines()
