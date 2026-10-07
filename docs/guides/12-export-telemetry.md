# Export Telemetry to Any Backend

Run the community extenders (`OtelExtender` for spans, `OpenLineageExtender` for lineage events) against any backend. mloda contains no cloud code and sends nothing on its own: it emits through the provider or client you configure. The community extenders emit spans and OpenLineage events only, no log records. Extender internals are in [Create an Extender Plugin](11-create-extender.md). Select step spans by `mloda.operation.name`, not by name (see [Selecting OTel spans](11-create-extender.md#selecting-otel-spans)).

## Install

```bash
pip install mloda-community-otel opentelemetry-sdk opentelemetry-exporter-otlp
pip install mloda-community-openlineage  # lineage events, optional
```

`mloda-community[otel]` and `mloda-community[openlineage]` install the same extenders; core mloda has no `otel` extra. `mloda-community-otel` depends on `opentelemetry-api` only, so nothing is exported until an SDK provider is set.

## Wire a tracer provider

Keep the provider setup in an importable module (not `__main__`), so `MULTIPROCESSING` workers can run it too:

```python
# telemetry.py
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def install_tracer_provider() -> None:
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
```

```python
# main.py
from mloda.community.extenders.otel import OtelExtender
from mloda.user import mloda

from telemetry import install_tracer_provider

if __name__ == "__main__":
    install_tracer_provider()
    results = mloda.run_all(["my_feature"], function_extender={OtelExtender(use_sdk_defaults=True)})
```

`use_sdk_defaults=True` uses the globally set provider. Injecting `OtelExtender(tracer_provider=provider)` works too, but an SDK provider cannot be pickled, so it does not reach `MULTIPROCESSING` workers. Without either, the extender is inert and warns once; see [Sink Resolution](11-create-extender.md#sink-resolution). For OTLP over HTTP (port 4318, which many SaaS backends require), import `OTLPSpanExporter` from `opentelemetry.exporter.otlp.proto.http.trace_exporter` instead; `OTEL_EXPORTER_OTLP_PROTOCOL` does not switch an exporter you construct yourself. A vendor distro that installs its own global provider (for example `configure_azure_monitor()`) replaces `install_tracer_provider`, in `child_bootstrap` too.

## Export through a Collector

Send OTLP to an OpenTelemetry Collector (gRPC 4317, HTTP 4318) and let the Collector fan out to your backends. Endpoint, TLS and auth are standard OTLP exporter settings, not mloda settings:

| Variable | Purpose |
|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Collector or backend address |
| `OTEL_EXPORTER_OTLP_INSECURE` | `true` for a plaintext gRPC Collector (or use an `http://` endpoint) |
| `OTEL_EXPORTER_OTLP_HEADERS` | Auth, e.g. `x-api-key=...` for SaaS backends |
| `OTEL_EXPORTER_OTLP_CERTIFICATE`, `OTEL_EXPORTER_OTLP_CLIENT_KEY`, `OTEL_EXPORTER_OTLP_CLIENT_CERTIFICATE` | TLS and mTLS |
| `OTEL_SERVICE_NAME` | Service name on every span |
| `OTEL_TRACES_SAMPLER`, `OTEL_TRACES_SAMPLER_ARG` | Head sampling |

| Backend | Path | Watch for |
|---|---|---|
| AWS | ADOT Collector to X-Ray | X-Ray keeps traces 30 days |
| Google Cloud | Collector `googlecloud` exporter to Cloud Trace | Cloud Trace keeps traces 30 days |
| Azure | Azure Monitor OpenTelemetry distro, or the Collector `azuremonitor` exporter | The distro samples in the SDK; retention is set on the Log Analytics workspace |
| Self-hosted | Grafana Tempo, Jaeger, SigNoz over OTLP | Retention is yours; the only path with no external egress |
| SaaS | OTLP with a header token | Check each vendor's retention window and default sampling |

Switching backends changes only exporter or Collector configuration.

## Sampling and retention

Trace backends sample and expire spans, so a span is not a retained record.

- **Keep every run:** `OTEL_TRACES_SAMPLER=always_on` (read by `TracerProvider()` at construction). It also overrides an upstream caller's decision. The default `parentbased_always_on` follows the parent, so a run under an unsampled caller span or carrier is dropped with all its steps, and no Collector policy can bring back spans a head sampler dropped.
- **Keep failures with tail sampling:** a Collector `tail_sampling` policy on status `ERROR` keeps failed runs, failed plans and failed steps (they end with status `ERROR`, usually with `error.type`). Set `decision_wait` above your longest run, since the `mloda.run` root ends last, and add a baseline policy unless you mean to drop every successful run. Tail sampling needs every span of a trace on one Collector instance; `MULTIPROCESSING` workers export one trace from several processes, so put the `loadbalancing` exporter (`routing_key: traceID`) in front of several Collector replicas.
- **`trace_scope="plan"`:** head sampling keeps or drops all runs of a plan together.
- **Long retention belongs to the Collector, not mloda:** route the same receiver into a second traces pipeline, with no sampling processor, that exports to object storage with bucket immutability (S3 Object Lock, GCS Bucket Lock, Azure immutable storage). OpenLineage events on a durable transport are the dataset-centric record.
- **Audit records (enterprise, licensed):** `AuditExtender` with `OtelLogAuditSink` (`mloda-enterprise[otel]`) emits allow and deny decisions as log records, the long-retention log channel. It needs an SDK `LoggerProvider` (`set_logger_provider` with a `BatchLogRecordProcessor(OTLPLogExporter())`), installed in `child_bootstrap` as well under `MULTIPROCESSING`; the sealed NDJSON log stays the retained record. See [Verified run context](11-create-extender.md#verified-run-context).

`tail_sampling`, `redaction` and `loadbalancing` ship in the Collector contrib distribution (`otelcol-contrib`), not in the core `otelcol`; vendor distributions such as ADOT have their own component lists.

```yaml
processors:
  tail_sampling:
    decision_wait: 300s  # longer than your longest run
    policies:
      - name: keep-errors
        type: status_code
        status_code: {status_codes: [ERROR]}
      - name: baseline
        type: probabilistic
        probabilistic: {sampling_percentage: 10}

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [tail_sampling, redaction, batch]
      exporters: [otlp]
```

## Privacy defaults

Spans are metadata only by default.

- **Never recorded:** feature values and row data; exception messages (spans and extender warnings carry only the exception type as `error.type`); baggage (carriers carry `traceparent` and `tracestate`, never baggage, inbound and outbound).
- **Recorded, review before export:** feature, feature group (also in span names) and compute framework names and versions; plugin versions; row counts; `mloda.data_access.identity` and `mloda.data_access.format` on load spans; `mloda.join.keys` (column names); `mloda.declared.*` (scalars a feature group author declares); run, plan and step ids; the worker index. By default the data-access identity is a URI's scheme, host and path, a local path, or mapping keys (never mapping values), but a reader can override `data_access_identity`, and a path can still name a customer, so review custom readers.
- **Content previews are opt-in:** `OtelExtender(capture_content=True, mask=...)` records a bounded, scrubbed preview of the masked result as `mloda.content.preview`; `capture_content=True` without a `mask` raises. `MLODA_OTEL_TRACE_CONTENT=true` (or `1`) turns capture on for extenders left at the default `capture_content=None`, still only with a `mask`; without one it warns once and records nothing. An explicit `capture_content=False` wins over the env var.
- **Identity:** `OtelExtender` records no tenant, project or principal. The enterprise `OtelLogAuditSink` pseudonymises only the principal (`user.hash`, only with a `user_hash_key`) and exports `mloda.tenant.id` and `mloda.project.id` as given; see [Verified run context](11-create-extender.md#verified-run-context).
- **OpenLineage:** events carry job names, input feature names, output schema fields (names and types) and load dataset names (derived from core's data-access identity; the `dataSource` facet carries the identity itself, as on load spans).
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

`OpenLineageExtender(use_sdk_defaults=True)` uses openlineage-python's own configuration, with no vendor code in mloda: `OPENLINEAGE_URL` (plus `OPENLINEAGE_API_KEY`), or `OPENLINEAGE_CONFIG` / `openlineage.yml` / `OPENLINEAGE__TRANSPORT__*` for any openlineage-python transport (`http`, `async_http`, `kafka` with `openlineage-python[kafka]`, cloud transports, `composite`). With nothing configured it falls back to the console transport, which logs every full event at INFO; set `OPENLINEAGE_DISABLED=true` to turn it off. Emission runs on the calculation thread, so prefer `async_http` or a short timeout. See [Sink Resolution](11-create-extender.md#sink-resolution) and [Emitting on the calculation thread](11-create-extender.md#emitting-on-the-calculation-thread).

## Traces and lineage

With `OtelExtender` outside `OpenLineageExtender` (what the default priorities give, 100 and 110), each step's calculate span is current while OpenLineage emits, so the step's RunEvents carry a run facet `mlodaTrace` (`traceId`, `spanId`) pointing at that span. Join them through the span attribute `mloda.step.run_id`, which equals the step RunEvent's runId. The root run has no such facet; its runId equals `mloda.run.id`. No facet is added when no recording span is current (no provider, a non-recording span, `opentelemetry` not installed, or OpenLineage priority below `OtelExtender`).

## Multiprocessing, threads and asyncio

- **`MULTIPROCESSING`:** each spawned worker needs its own provider. Add `parallelization_modes={ParallelizationMode.MULTIPROCESSING}` and `child_bootstrap=install_tracer_provider` (picklable, from an importable module) to `run_all`, `stream_all`, `run` or `stream_run`, and use `OtelExtender(use_sdk_defaults=True)`. Worker spans join the run's trace on their own. `OpenLineageExtender(use_sdk_defaults=True)` builds its client per worker; an injected client survives only if it pickles.
- **Flush on worker exit:** core calls each extender's `close()` when a worker exits, within `graceful_shutdown_timeout` (default 2s, shared by the worker's extenders). With a `BatchSpanProcessor`, raise it together with the extender's `close_timeout` attribute (default 1s, e.g. `extender.close_timeout = 5.0`) so the batch drains. Details in [Pickle Compatibility](11-create-extender.md#pickle-compatibility).
- **Continue a trace from another process:** pass a W3C trace-context `carrier` to `run_all`, `stream_all`, `run` or `stream_run`. The run's root span becomes its child (with `trace_scope="plan"` the plan span stays the parent and the carrier becomes a span link). `inject_carrier()` from `mloda.community.extenders.otel.otel_multiprocessing` builds one from the current context without baggage.
- **`THREADING`:** nothing extra; steps find their run's root span by run id.
- **asyncio:** the active span and `verified_context` are context variables. `asyncio.to_thread(...)` copies them; `loop.run_in_executor(...)` does not, so use `loop.run_in_executor(None, contextvars.copy_context().run, fn)`.
