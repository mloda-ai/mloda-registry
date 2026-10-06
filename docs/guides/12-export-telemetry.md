# Export Telemetry to Any Backend

Run the community extenders (`OtelExtender` for spans, `OpenLineageExtender` for lineage events) against any backend. mloda contains no cloud code and sends nothing on its own: it emits through the provider or client you configure. Extender internals are in [Create an Extender Plugin](11-create-extender.md).

## Install

```bash
pip install mloda-community-otel opentelemetry-sdk opentelemetry-exporter-otlp
pip install mloda-community-openlineage  # lineage events, optional
```

`mloda-community[otel]` and `mloda-community[openlineage]` install the same extenders; core mloda has no `otel` extra. `mloda-community-otel` depends on `opentelemetry-api` only, so nothing is exported until an SDK provider is set.

## Wire a tracer provider

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from mloda.community.extenders.otel import OtelExtender
from mloda.user import mloda


def install_tracer_provider() -> None:
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)


install_tracer_provider()
results = mloda.run_all(["my_feature"], function_extender={OtelExtender(use_sdk_defaults=True)})
```

`use_sdk_defaults=True` uses the globally set provider. Injecting `OtelExtender(tracer_provider=provider)` works too, but an SDK provider cannot be pickled, so it does not reach `MULTIPROCESSING` workers (see [Multiprocessing, threads and asyncio](#multiprocessing-threads-and-asyncio)). Without either, the extender is inert and warns once; see [Sink Resolution](11-create-extender.md#sink-resolution).

## Export through a Collector

Send OTLP to an OpenTelemetry Collector (gRPC 4317, HTTP 4318) and let the Collector fan out to your backends. Endpoint, TLS and auth are standard OTLP exporter settings, not mloda settings:

| Variable | Purpose |
|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Collector or backend address |
| `OTEL_EXPORTER_OTLP_HEADERS` | Auth, e.g. `x-api-key=...` for SaaS backends |
| `OTEL_EXPORTER_OTLP_CERTIFICATE`, `OTEL_EXPORTER_OTLP_CLIENT_KEY`, `OTEL_EXPORTER_OTLP_CLIENT_CERTIFICATE` | TLS and mTLS |
| `OTEL_SERVICE_NAME` | Service name on every span |
| `OTEL_TRACES_SAMPLER`, `OTEL_TRACES_SAMPLER_ARG` | Head sampling |

| Backend | Path | Watch for |
|---|---|---|
| AWS | ADOT Collector to X-Ray | X-Ray keeps traces 30 days |
| Google Cloud | Collector with the Google Cloud exporter to Cloud Trace | Cloud Trace keeps traces 30 days |
| Azure | Azure Monitor OpenTelemetry distro or a Collector exporter | Retention and adaptive sampling are workspace settings |
| Self-hosted | Grafana Tempo, Jaeger, SigNoz over OTLP | Retention is yours; the only path with no external egress |
| SaaS | OTLP with a header token | Check each vendor's retention window and default sampling |

Switching backends changes only exporter or Collector configuration.

## Sampling and retention

Trace backends sample and expire spans, so a span is not a retained record.

- **Keep every run:** `OTEL_TRACES_SAMPLER=always_on`. The default `parentbased_always_on` follows the parent's decision, so a run under an unsampled caller span or carrier is dropped with all its steps.
- **Keep failures with tail sampling:** a Collector `tail_sampling` policy on status `ERROR` keeps failed runs and failed plans (both end with status `ERROR` and carry `error.type`). Tail sampling needs every span of a trace on one Collector instance; `MULTIPROCESSING` workers export one trace from several processes, so put a trace-ID load balancer in front of several Collector replicas.
- **`trace_scope="plan"`:** head sampling keeps or drops all runs of a plan together.
- **Long retention belongs to the Collector, not mloda:** add a second exporter to the traces pipeline that writes to object storage with bucket immutability (S3 Object Lock, GCS Bucket Lock, Azure immutable storage). OpenLineage events on a durable transport are the dataset-centric record.
- **Audit records (enterprise):** `AuditExtender` with `OtelLogAuditSink` (`mloda-enterprise[otel]`, licensed) emits allow and deny decisions as log records; it needs an SDK `LoggerProvider` and exporter, and the sealed NDJSON log stays the retained record. See [Verified run context](11-create-extender.md#verified-run-context).

```yaml
processors:
  tail_sampling:
    policies:
      - name: keep-errors
        type: status_code
        status_code: {status_codes: [ERROR]}
```

## Privacy defaults

Spans are metadata only by default.

- **Never recorded:** feature values and row data; exception messages (spans and extender warnings carry only the exception type as `error.type`); baggage (carriers carry `traceparent` only, inbound and outbound).
- **Recorded, review before export:** feature, feature group and compute framework names; `mloda.data_access.identity` on load spans (a URI's scheme, host and path, a local path, or mapping keys, never mapping values); `mloda.join.keys` (column names); `mloda.declared.*` (scalars a feature group author declares); run, plan and step ids.
- **Content previews are opt-in:** `OtelExtender(capture_content=True, mask=...)` records a bounded, scrubbed preview of the masked result as `mloda.content.preview`; `capture_content=True` without a `mask` raises. `MLODA_OTEL_TRACE_CONTENT=true` (or `1`) turns capture on for extenders left at the default `capture_content=None`, still only with a `mask`; without one it warns once and records nothing. An explicit `capture_content=False` wins over the env var.
- **Pseudonymous identity:** `OtelExtender` records no tenant, project or principal. The enterprise `OtelLogAuditSink` adds `user.hash` only with a `user_hash_key`; see [Verified run context](11-create-extender.md#verified-run-context).
- **OpenLineage:** load dataset names are core's `data_access_identity`, as on load spans.
- **Your application's own logs** are outside these guarantees: a logging bridge to OTel ships whatever your code and libraries log.

The Collector `redaction` processor is a second line of defence. Keep `allow_all_keys: true`, otherwise every `mloda.*` key not listed is dropped:

```yaml
processors:
  redaction:
    allow_all_keys: true
    blocked_values:
      - "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"  # email addresses
```

It sees what passes through the Collector only; OpenLineage events go straight to their transport.

## OpenLineage

`OpenLineageExtender(use_sdk_defaults=True)` uses openlineage-python's own configuration, with no vendor code in mloda: `OPENLINEAGE_URL` (plus `OPENLINEAGE_API_KEY`), or `OPENLINEAGE_CONFIG` / `openlineage.yml` / `OPENLINEAGE__TRANSPORT__*` for the `http`, `async_http` and `kafka` transports. Emission runs on the calculation thread, so prefer `async_http` or a short timeout. See [Sink Resolution](11-create-extender.md#sink-resolution) and [Emitting on the calculation thread](11-create-extender.md#emitting-on-the-calculation-thread).

## Multiprocessing, threads and asyncio

- **`MULTIPROCESSING`:** each spawned worker needs its own provider. Pass a picklable, module-level `child_bootstrap` (such as `install_tracer_provider` above) to `run_all`, `stream_all`, `run` or `stream_run`, and use `OtelExtender(use_sdk_defaults=True)`. Worker spans join the run's trace on their own. `OpenLineageExtender(use_sdk_defaults=True)` builds its client per worker; an injected client survives only if it pickles.
- **Flush on worker exit:** core calls each extender's `close()` when a worker exits, within `graceful_shutdown_timeout` (default 2s, shared by the worker's extenders). With a `BatchSpanProcessor`, raise it together with the extender's `close_timeout` attribute (default 1s) so the batch drains. Details in [Pickle Compatibility](11-create-extender.md#pickle-compatibility).
- **Continue a trace from another process:** pass `carrier={"traceparent": ...}` (W3C trace context) to `run_all`, `stream_all`, `run` or `stream_run`; the run's root span becomes its child. Build it with `TraceContextTextMapPropagator().inject(carrier)` so no baggage rides along.
- **`THREADING`:** nothing extra; steps find their run's root span by run id.
- **asyncio:** `asyncio.to_thread(...)` copies the caller's context. `loop.run_in_executor(...)` does not, which drops the active span and also `verified_context`; use `loop.run_in_executor(None, contextvars.copy_context().run, fn)`.
