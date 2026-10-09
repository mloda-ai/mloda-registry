# Local OTel + OpenLineage demo

Runs the mloda OTel and OpenLineage extenders against a local stack: Collector, Tempo (traces), Prometheus (metrics),
Grafana and Marquez (lineage). Not part of any wheel.

## Start

```bash
cd examples/otel_demo
docker compose up -d
```

Run from the repo root (the pinned exporter package matches the locked SDK, so the lock does not change):

```bash
uv run --with opentelemetry-exporter-otlp-proto-http==1.45.0 python examples/otel_demo/pipeline.py --source csv
uv run --with opentelemetry-exporter-otlp-proto-http==1.45.0 python examples/otel_demo/pipeline.py --source parquet
uv run --with opentelemetry-exporter-otlp-proto-http==1.45.0 python examples/otel_demo/pipeline.py --source memory
uv run --with opentelemetry-exporter-otlp-proto-http==1.45.0 python examples/otel_demo/pipeline.py --fail
```

`--fail` makes every other run fail so the failed step panel has data. `--runs` and `--interval` control the loop.

## Where to click

- Grafana (http://localhost:3000): Explore, Tempo, search by service `mloda-otel-demo`. TraceQL example:
  `{ span.mloda.data_access.format = "ParquetReader" }`. The `mloda steps` dashboard shows p95 step duration and the
  failed step share.
- Marquez UI (http://localhost:3001): jobs, datasets and runs of the OpenLineage events.

## Correlation

- OpenLineage root `runId` equals the span attribute `mloda.run.id`.
- OpenLineage step `runId` equals the span attribute `mloda.step.run_id`.
- Step events carry the `mlodaTrace` run facet.

## Offline

Zero external egress once the images are pulled and the uv `--with` package is cached.

## Limits

- Grafana has no Marquez data source, so the dashboard links to the Marquez UI.
- Marquez images are amd64 only (emulated on Apple Silicon, slow start).
- Marquez 0.51.1 models only `run` and `job` of the OpenLineage `parent` facet, so its `root` (set via
  `parent_id`/`root_parent_id`) is not used for its run hierarchy. Not relevant here unless a parent is set.
- Metrics come from one process; each process gets its own `service.instance.id` series.
- Field names, paths and feature group names reach every backend (see the privacy section of the export guide).

## Stop

```bash
docker compose down -v
```
