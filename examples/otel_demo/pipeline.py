"""Run the demo pipeline in a loop and export OTLP telemetry and OpenLineage events to the local stack.

uv run --with opentelemetry-exporter-otlp-proto-http==1.45.0 python examples/otel_demo/pipeline.py --source csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import demo_features  # noqa: E402

logger = logging.getLogger("otel_demo")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=demo_features.SOURCES, default="csv")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--interval", type=float, default=3.0, help="seconds between runs")
    parser.add_argument("--fail", action="store_true", help="make every other run fail")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    os.environ.setdefault("OTEL_SERVICE_NAME", "mloda-otel-demo")
    # Explicit transport config (the simple OPENLINEAGE_URL form has no timeout); short timeout when Marquez is down.
    os.environ.setdefault("OPENLINEAGE__TRANSPORT__TYPE", "http")
    os.environ.setdefault("OPENLINEAGE__TRANSPORT__URL", os.environ.get("OPENLINEAGE_URL", "http://localhost:5002"))
    os.environ.setdefault("OPENLINEAGE__TRANSPORT__TIMEOUT", "5")

    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import Counter, Histogram, MeterProvider
    from opentelemetry.sdk.metrics.export import AggregationTemporality, PeriodicExportingMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    # Delta temporality: the Collector turns it back into cumulative (deltatocumulative processor).
    delta = {Counter: AggregationTemporality.DELTA, Histogram: AggregationTemporality.DELTA}
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tracer_provider)
    reader = PeriodicExportingMetricReader(OTLPMetricExporter(preferred_temporality=delta), export_interval_millis=2000)
    meter_provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(meter_provider)

    from mloda.community.extenders.openlineage.openlineage_extender import OpenLineageExtender
    from mloda.community.extenders.otel.otel_extender import OtelExtender
    from mloda.community.extenders.otel.otel_metrics_extender import OtelMetricsExtender

    extenders = {
        OtelExtender(use_sdk_defaults=True),
        OtelMetricsExtender(use_sdk_defaults=True),
        OpenLineageExtender(use_sdk_defaults=True),
    }
    data_dir = demo_features.default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        for i in range(args.runs):
            fail = args.fail and i % 2 == 1
            try:
                values = demo_features.run_pipeline(args.source, extenders, data_dir, fail=fail)
                logger.info("run %d (%s): %s", i, args.source, values)
            except ValueError as exc:
                logger.warning("run %d failed: %s", i, exc)
            time.sleep(args.interval)
    finally:
        tracer_provider.force_flush()
        meter_provider.force_flush()
        tracer_provider.shutdown()
        meter_provider.shutdown()


if __name__ == "__main__":
    main()
