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
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from mloda.community.extenders.otel import OtelExtender
from mloda.testing.extenders.otel import FileSpanExporter
from mloda.testing.extenders.runners import expected_value_int, run_value_int


class _InstallRealTracerProviderBootstrap:
    """Picklable child_bootstrap (a plain callable defined at module level, not a closure): installs a
    real SDK TracerProvider, exporting to marker_path, as the process-global provider inside a spawned
    MULTIPROCESSING worker, before that worker processes its first command.

    OtelExtender(use_sdk_defaults=True) (no injected tracer_provider) then resolves this ambiently via
    opentelemetry.trace.get_tracer_provider(), the process-global provider this bootstrap just set.
    """

    def __init__(self, marker_path: Path) -> None:
        self._marker_path = marker_path

    def __call__(self) -> None:
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(FileSpanExporter(self._marker_path)))
        trace.set_tracer_provider(provider)


def test_child_bootstrap_installed_provider_emits_a_span_inside_the_spawned_worker(
    tmp_path: Path, flight_server: ParallelRunnerFlightServer
) -> None:
    marker_path = tmp_path / "otel_multiprocessing_spans.txt"
    bootstrap = _InstallRealTracerProviderBootstrap(marker_path)

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
    assert "mloda.calculate" in span_names, span_names
