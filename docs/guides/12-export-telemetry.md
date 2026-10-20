# Export Telemetry to Any Backend

Run the community extenders (`OtelExtender` for spans, `OtelMetricsExtender` for metrics, `OpenLineageExtender` for lineage events) against any backend. mloda contains no cloud code and sends nothing on its own: it emits through the provider or client you configure. The community extenders emit spans, metrics and OpenLineage events only, no log records. Extender internals are in [Create an Extender Plugin](11-create-extender.md). Select step spans by `mloda.operation.name`, not by name (see [Selecting OTel spans](11-create-extender.md#selecting-otel-spans)).

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

## Metrics

`OtelMetricsExtender` records four metrics through the OTel metrics API, independent of `OtelExtender`. Add a meter provider next to the tracer provider in `telemetry.py`:

```python
from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader


def install_meter_provider() -> None:
    reader = PeriodicExportingMetricReader(OTLPMetricExporter())
    metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
```

`OtelMetricsExtender(use_sdk_defaults=True)` picks up the global meter provider. `OtelMetricsExtender(meter_provider=provider)` injects one (it does not pickle, like `tracer_provider`). With no SDK meter provider nothing is recorded and it warns once; see [Sink Resolution](11-create-extender.md#sink-resolution).

| Instrument | Kind, unit | Attributes |
|---|---|---|
| `mloda.run.duration` | Histogram, `s` | `mloda.run.status`, `error.type` (failed runs), `mloda.cancel.type` (cancelled runs) |
| `mloda.step.duration` | Histogram, `s` | `mloda.operation.name`, `mloda.feature_group.name`, `mloda.compute_framework.name`, `error.type` (failed steps), `mloda.cancel.type` (cancelled steps) |
| `mloda.step.rows.in` | Counter, `{row}` | `mloda.operation.name`, `mloda.feature_group.name`, `mloda.compute_framework.name` |
| `mloda.step.rows.out` | Counter, `{row}` | same as `mloda.step.rows.in` |

- **Outcomes:** no separate counters; a histogram's count is the outcome counter, split by `mloda.run.status`, `error.type` or `mloda.cancel.type`. Calls without a hook context are not counted.
- **`error.type`:** `module.qualname` on runs and steps (as on spans), so a builtin reads `builtins.ValueError`. Before mloda 0.16.0, runs carried the bare class name.
- **Cancelled:** an interrupt (a `BaseException` that is not an `Exception`, such as `SystemExit` from SIGTERM or `KeyboardInterrupt`) is not a failure: it sets `mloda.cancel.type` (`builtins.SystemExit`) instead of `error.type`, so `error_type!=""` queries leave out runs and steps a rollout or eviction stopped.
- **Rows:** counted on success only, for calculate (in and out) and load (out). Validate steps record no rows.
- **Durations:** a step duration includes inner extenders (with default priorities `OtelExtender` wraps outside `OtelMetricsExtender`, which wraps outside `OpenLineageExtender`, so it includes OpenLineage emission) and a calculate duration includes a nested load, so do not sum durations across operations.
- **Attributes:** only those in the table; unset ones (for example the feature group on a join) are omitted. Never recorded on metrics: feature names, run, plan and step ids, worker index, data-access identity and format, join keys and type, declared attributes, result attributes, plugin versions, tenant, project, principal.
- **Failures:** recording is best effort; a post-call recording error is logged once at WARNING with its type only and never changes the step's result or exception. A failure resolving the meter follows `raise_on_error` like any extender.

Prometheus names after OTLP translation, p95 step duration per feature group and the share of failed steps:

```promql
histogram_quantile(0.95, sum by (le, mloda_feature_group_name) (rate(mloda_step_duration_seconds_bucket[5m])))

sum by (mloda_feature_group_name) (rate(mloda_step_duration_seconds_count{error_type!=""}[5m]))
  / sum by (mloda_feature_group_name) (rate(mloda_step_duration_seconds_count[5m]))

sum by (mloda_cancel_type) (rate(mloda_step_duration_seconds_count{mloda_cancel_type!=""}[5m]))
```

### Metrics from short-lived processes

`MULTIPROCESSING` workers and Kubernetes Job or CronJob runs each export from their own process with per-process resource attributes (by default a random `service.instance.id`; with the Operator or `k8sattributes` also pod, job and node names). Each process then gets its own short series, and the queries above return 0 or nothing under either temporality. To aggregate, export deltas (`OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=delta` in the process environment; under `MULTIPROCESSING` the parent's, which its workers inherit) and merge them in the Collector (`transform` and `delta_to_cumulative`, both in `otelcol-contrib`):

```yaml
processors:
  transform/mloda_merge:
    metric_statements:
      - context: resource
        statements:
          # keep only attributes that are the same on every run
          - keep_keys(resource.attributes, ["service.name", "service.namespace"]) where resource.attributes["service.name"] == "my-pipeline"
      - context: datapoint
        statements:
          # any fixed time before the first point
          - set(datapoint.start_time, Time("2026-01-01T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ")) where resource.attributes["service.name"] == "my-pipeline"
          - set(datapoint.time, Now()) where resource.attributes["service.name"] == "my-pipeline"
  delta_to_cumulative:
    max_stale: 25h  # above the longest gap between runs
service:
  pipelines:
    metrics:
      processors: [transform/mloda_merge, delta_to_cumulative, batch]
```

- Give the pipeline a `service.name` (`OTEL_SERVICE_NAME`) no other process uses: every process with that name is merged, so per-process values are gone and point times are the Collector's arrival times. Only counters and histograms (all four mloda metrics) merge correctly; keep gauges and up-down counters under another service.
- `keep_keys` drops every attribute it does not list; deleting only `service.instance.id` is not enough on Kubernetes, where per-run pod, job and node names keep runs apart and the exporter overwrites them on one series. The `prometheus` exporter labels series only by `service.namespace`, `service.name` and `service.instance.id`, so separate pipelines you must not merge (dev and prod) by those; another kept attribute becomes a label only with the exporter's `resource_to_telemetry_conversion` enabled. If `k8sattributes` supplies a kept attribute, run it before the transform, in the first tier when a `loadbalancing` exporter is in front.
- The restamp stops most out-of-order drops between concurrent processes; the Collector's own `deltatocumulative` datapoints counter, with an `error` attribute, counts the rest.
- One Collector instance must receive all of the pipeline's metrics: a central gateway, not a sidecar per pod (with several replicas, put a `loadbalancing` exporter with `routing_key: service` in front). Its state is in memory: after a restart, or once a series is idle longer than `max_stale`, the series restarts from the next delta, and Prometheus sees a reset only if the value drops, so totals across it can be wrong. Keep `max_stale`, and the `prometheus` exporter's `metric_expiration`, above the longest gap between runs. `delta_to_cumulative` is alpha.
- For scheduled runs, query with a range of at least the schedule interval (`increase(...[1d])` for a daily job); a `[5m]` range returns 0 between runs.

## Export through a Collector

Send OTLP to an OpenTelemetry Collector (gRPC 4317, HTTP 4318) and let the Collector fan out to your backends. Endpoint, TLS and auth are standard OTLP exporter settings, not mloda settings:

| Variable | Purpose |
|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Collector or backend address |
| `OTEL_EXPORTER_OTLP_INSECURE` | `true` for a plaintext gRPC Collector (or use an `http://` endpoint) |
| `OTEL_EXPORTER_OTLP_HEADERS` | Auth, e.g. `x-api-key=...` for SaaS backends |
| `OTEL_EXPORTER_OTLP_CERTIFICATE`, `OTEL_EXPORTER_OTLP_CLIENT_KEY`, `OTEL_EXPORTER_OTLP_CLIENT_CERTIFICATE` | TLS and mTLS |
| `OTEL_SERVICE_NAME` | Service name on every span and metric |
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
- **Keep failures with tail sampling:** a Collector `tail_sampling` policy on status `ERROR` keeps failed runs, failed plans and failed steps (they end with status `ERROR`, usually with `error.type`); a failing `explain` or `diagnose` never ends in `ERROR`. Cancelled runs and steps do not end with `ERROR`; to keep them too, add a policy on the attribute `mloda.run.status` with value `cancelled`. Set `decision_wait` above your longest run, since the `mloda.run` root ends last, and add a baseline policy unless you mean to drop every successful run. Tail sampling needs every span of a trace on one Collector instance; `MULTIPROCESSING` workers export one trace from several processes, so put the `loadbalancing` exporter (`routing_key: traceID`) in front of several Collector replicas.
- **`trace_scope="plan"`:** head sampling keeps or drops all runs of a plan together.
- **Long retention belongs to the Collector, not mloda:** route the same receiver into a second traces pipeline, with no sampling processor, that exports to object storage with bucket immutability (S3 Object Lock, GCS Bucket Lock, Azure immutable storage). OpenLineage events on a durable transport are the dataset-centric record.
- **Audit records (enterprise, license agreement for production use):** `AuditExtender` with `OtelLogAuditSink` (`mloda-enterprise[otel]`) emits allow and deny decisions as log records, the long-retention log channel. It needs an SDK `LoggerProvider` (`set_logger_provider` with a `BatchLogRecordProcessor(OTLPLogExporter())`), installed in `child_bootstrap` as well under `MULTIPROCESSING`; the sealed NDJSON log stays the retained record. See [Verified run context](11-create-extender.md#verified-run-context).

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
    metrics:
      receivers: [otlp]
      processors: [redaction, batch]
      exporters: [otlp]
```

## Privacy defaults

Spans and metrics are metadata only by default; metrics carry a small subset (see [Metrics](#metrics)), and the feature group and compute framework class names reach metrics backends.

- **Never recorded:** feature values and row data (except what a plugin itself publishes as `mloda.result.*`); exception messages (spans and extender warnings carry only the exception type, as `error.type` or `mloda.cancel.type`); baggage (carriers carry `traceparent` and `tracestate`, never baggage, inbound and outbound).
- **Recorded, review before export:** feature, feature group (also in span names) and compute framework names and versions; plugin versions; row counts; `mloda.data_access.identity` and `mloda.data_access.format` on load spans; `mloda.join.keys` (column names); `mloda.declared.*` (scalars a feature group author declares, also recorded as the `mlodaDeclared` facet and in the audit record); `mloda.result.*` (data-dependent scalars a feature group or reader publishes; string values are scrubbed of credentials like the content preview but not masked, so never publish row values or credentials); `mloda.data.classification` (with `classification_undeclared`); run, plan and step ids; the worker index. By default the data-access identity is a URI's scheme, host and path, a local path, or mapping keys (never mapping values), but a reader can override `data_access_identity`, and a path can still name a customer, so review custom readers.
- **Content previews are opt-in:** `OtelExtender(capture_content=True, mask=...)` records a bounded, scrubbed preview of the masked result as `mloda.content.preview`; `capture_content=True` without a `mask` raises. `MLODA_OTEL_TRACE_CONTENT=true` (or `1`) turns capture on for extenders left at the default `capture_content=None`, still only with a `mask`; without one it warns once and records nothing. An explicit `capture_content=False` wins over the env var.
- **Identity:** `OtelExtender` records no tenant, project or principal. The enterprise `OtelLogAuditSink` pseudonymises only the principal (`user.hash`, only with a `user_hash_key`) and exports `mloda.tenant.id` and `mloda.project.id` as given; see [Verified run context](11-create-extender.md#verified-run-context).
- **OpenLineage:** events carry job names, input feature names, output schema fields (names and types) and load dataset names (derived from core's data-access identity; the `dataSource` facet carries the identity itself, as on load spans; the `mlodaDeclared` facet carries the scalars a reader declares on input datasets and a feature group declares on output datasets); with `classification_undeclared`, output datasets also carry a `mloda.data.classification` tag; `ownership` facets carry the owner names a feature group declares (`mloda.owner`) and the deployer passes as `dataset_owners`; step runs carry the feature group and plugin versions and the compute framework name (`mlodaStep`), the root run the mloda and `mloda-community-openlineage` versions (`processingEngine`), and output datasets the step's row count (`outputStatistics`), with no opt-out.
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

### Local run record without a backend

To keep a run record without Marquez or Tempo, write the events to a local file with openlineage-python's `file` transport:

```yaml
# openlineage.yml (or point OPENLINEAGE_CONFIG at it)
transport:
  type: file
  log_file_path: openlineage.ndjson  # the directory must exist
  append: true
```

```python
OpenLineageExtender(use_sdk_defaults=True)
```

Each line is one event. Per run:

- **Root run** (job `mloda.run_all`, runId = `mloda.run.id`): `mlodaPlan` (plan id, structure hash, origin, and `requestedFeatures`, the sorted names the run's compute steps were asked for) and `processingEngine` (`name: mloda`, the mloda version, and the `mloda-community-openlineage` version as `openlineageAdapterVersion`). The job name stays `mloda.run_all` whatever the origin.
- **Each calculate step** (job = feature group): `mlodaStep` (`featureGroupVersion`, `pluginVersion`, `computeFramework`, and `featureNames`, the step's sorted feature names; a field core does not know is omitted) and `parent` (the root run).
- **Inputs:** loaded datasets with `dataSource` (core's data-access identity) and the step's declared input features.
- **COMPLETE outputs:** one per feature, with `schema` and, when the compute framework can count the result's rows (not for lazy frames), `outputStatistics.rowCount`. The count is the step's, repeated on every output of a multi-output step, so never sum it across outputs.

Notes:

- With `append: true`, runs accumulate in one file; group lines by the root runId (each step's `parent` facet points at it).
- Without `append`, every event goes to its own `<log_file_path>-<timestamp>.json` file.
- With `append: true` under `MULTIPROCESSING`, every worker appends to the same path. The `file` transport of openlineage-python writes each event in a single append write, so lines stay whole on a local POSIX file system. On NFS, Windows or an object store, leave `append` off. Without `append`, file names carry the local time to the microsecond, so if two workers emit in the same microsecond, one event overwrites the other.
- Diff two runs with the stdlib-only `mloda.community.extenders.openlineage.run_diff`:

  ```python
  from mloda.community.extenders.openlineage.run_diff import diff_run_events, read_runs

  runs = read_runs("openlineage.ndjson")
  print(diff_run_events(runs[-2], runs[-1]))
  ```

  It compares the structure hash, mloda version, outcome and, per step, the versions, compute framework, input features, loaded datasets, output schema, row count and outcome. Durations are listed as informational and never count as changes. Data-access identities appear as the events record them. Steps are keyed by feature group and feature names; duplicate steps of one feature group are told apart by compute framework, then order. Runs are comparable when the root job namespace and name match and, when both recorded them, the requested features match; otherwise they are reported as not comparable. `diff_run_events` takes one record per argument and raises `ValueError` for more. `read_runs` takes the file written with `append: true`, or the per-event files written without it, in name order: `read_runs(*sorted(glob.glob("openlineage.ndjson-*.json")))`. Keep `debug_mode` off, because its indented files are skipped.
- The events do not show how the plan is wired; keep a [plan lock](feature-group-patterns/10-testing-guide.md) next to the record for that.

## Traces and lineage

With `OtelExtender` outside `OpenLineageExtender` (what the default priorities give, 100 and 110), each step's calculate span is current while OpenLineage emits, so the step's RunEvents carry a run facet `mlodaTrace` (`traceId`, `spanId`) pointing at that step's own calculate span (matched by `mloda.step.run_id`). Join them through the span attribute `mloda.step.run_id`, which equals the step RunEvent's runId. Only calculate steps get it, not validation or root runs; a root run's runId equals `mloda.run.id`. No facet is added when the current span is not that step span: no provider, a non-recording span, `opentelemetry` not installed, or OpenLineage wrapping outside `OtelExtender` (lower priority), even if an application span is active.

## Run under an orchestrator

When a scheduler launches the process, attach the run to the scheduler's trace and lineage run. Both inputs come from the caller's environment, so trust them as much as `OPENLINEAGE_URL`: a spoofed parent id attaches runs under any job.

- **Trace:** `mloda.run_all(..., carrier=env_carrier())` (from `mloda.community.extenders.otel.otel_multiprocessing`) makes the run's root span a child of the span in `TRACEPARENT` (`TRACESTATE` is carried, `BAGGAGE` is never read). It returns `{}` when `TRACEPARENT` is unset or malformed, so the run starts its own trace. An unsampled `TRACEPARENT` (flags `00`) under the default `parentbased_always_on` sampler drops the whole run. With `trace_scope="plan"` the carrier is a span link, not the parent. Pass it only when no span is active; an active span you created is more specific.
- **Lineage:** `OpenLineageExtender(parent_id=..., root_parent_id=...)` (also `LineageFacetsExtender`) takes `{namespace}/{job_name}/{run_id}` with a UUID `run_id`, the format of the Airflow macros below; a namespace may contain `/`. The root run's START and terminal events then carry a `parent` facet with `root`, and each step's `parent` facet gains the same `root`. A missing `root_parent_id` defaults to the parent. Malformed explicit ids raise `ValueError`; `root_parent_id` needs `parent_id`. `parent_from_env=True` reads `OPENLINEAGE_PARENT_ID` and `OPENLINEAGE_ROOT_PARENT_ID` once, when the extender is built, unless `parent_id` is given. A malformed env value logs one warning naming the variable (never its value) and is ignored. Without the opt-in the environment is never read.

Airflow `KubernetesPodOperator` (`env_vars` is templated; the root macro needs `apache-airflow-providers-openlineage>=2.4.0`; set `TRACEPARENT` too only if your launcher provides one; on older providers omit the root variable and the root defaults to the parent):

```python
KubernetesPodOperator(
    ...,
    env_vars={
        "OPENLINEAGE_PARENT_ID": "{{ macros.OpenLineageProviderPlugin.lineage_parent_id(task_instance) }}",
        "OPENLINEAGE_ROOT_PARENT_ID": "{{ macros.OpenLineageProviderPlugin.lineage_root_parent_id(task_instance) }}",
    },
)
# in the container
OpenLineageExtender(use_sdk_defaults=True, parent_from_env=True)
```

Argo Workflows has no such macros and emits no OpenLineage events itself, so the parent shows as a placeholder job. Pick a fixed namespace, a stable job name (such as the WorkflowTemplate name) and use `{{workflow.uid}}`, a UUID, as the run id:

```yaml
env:
  - name: OPENLINEAGE_PARENT_ID
    value: "argo/my-workflow-template/{{workflow.uid}}"
```

Spark steps in the same pipeline read `spark.openlineage.parentJobNamespace`, `parentJobName` and `parentRunId` (and `rootParentJobNamespace`, `rootParentJobName`, `rootParentRunId`) at session start. Set them to the same orchestrator values so Spark's runs and the mloda run are siblings under the orchestrator task. Spark cannot point at an individual mloda step, since core mints the mloda run id inside `run_all`.

## Run on Kubernetes

### End a run on SIGTERM

Eviction, preemption and rollouts send SIGTERM. Python's default action exits without running `atexit` (as PID 1 it ignores SIGTERM until the SIGKILL at the end of the grace period), so the run keeps a START event with no terminal event and its open spans are lost. Opt in from the batch entrypoint (never from a library, and not in a server such as uvicorn, whose own graceful drain it would preempt):

```python
from mloda.community.extenders.shared.termination import install_sigterm_handler

if __name__ == "__main__":
    install_sigterm_handler(grace=25.0)  # below terminationGracePeriodSeconds (default 30)
    results = mloda.run_all(...)
```

On SIGTERM the handler calls any Python handler installed before it, then raises `SystemExit(143)` in the main thread. Open step spans and the `mloda.run` root end cancelled (status unset, `mloda.cancel.type` `builtins.SystemExit`, `mloda.run.status` `cancelled`), open OpenLineage step runs and the root run end with ABORT, and the exit flushes run as on a normal exit (OpenLineage self-built clients, up to 10s each; the SDK providers' own exit flush). After `grace` seconds a watchdog ends the process with `os._exit(143)` whatever is still running, so anything not yet flushed is lost. Notes:

- Run Python as PID 1 with an exec-form `CMD ["python", "main.py"]`, or under `tini`/`dumb-init`. A shell that stays PID 1 (a shell-form `CMD` with more than one command) does not forward SIGTERM.
- The grace period also covers `preStop`, and the handler only runs once the main thread is back in Python (a long C call delays it). The watchdog is the real cap; to fit the flush under `grace`, lower `OTEL_EXPORTER_OTLP_TIMEOUT` (default 10s per export; the batch processors ignore `OTEL_BSP_EXPORT_TIMEOUT`) and the OpenLineage `http` transport's timeout and retries.
- A second SIGTERM during the unwind is ignored.
- Full ABORT and flush cover `SYNC` runs. A `THREADING` run waits for the step in flight before it unwinds, so a long step reaches the watchdog first.
- Under `MULTIPROCESSING` the root run ends with ABORT, and core stops a worker that is still busy after `graceful_shutdown_timeout` (default 2s) with SIGTERM. Without a handler in the worker that step's terminal event and spans are lost. Install it there through `child_bootstrap`: `functools.partial(install_sigterm_handler, grace=10.0)`, or the same call at the end of the module-level bootstrap that installs your providers. The step then ends as in a `SYNC` run and the worker's extenders are closed. The cost: the handler answers every stop core sends a worker, on SIGTERM and when a run fails while another worker is busy. Core then waits for each busy worker to unwind, one after the other, instead of ending it at once, so size the graces as busy workers times the worker's `grace`, plus `graceful_shutdown_timeout`, below the parent's `grace`. A worker inside a long C call does not run the handler until the call returns; if the parent's watchdog fires first, the root run's ABORT is lost too. Signal the parent only, as Kubernetes does (`dumb-init` needs `--single-child`, since it signals the whole process group by default): a worker that receives SIGTERM itself exits. With the handler in that worker, its `SystemExit` reaches the parent and the run usually ends cancelled; without it, the run fails.
- `AuditExtender` flushes its sink but does not seal a run while terminating: the run stays pending for a `seal_ndjson_runs(..., older_than=...)` sweep, which seals it as `sealed_late` (see [Sealing and anchoring](11-create-extender.md#sealing-and-anchoring)). Schedule that sweep.

### OpenTelemetry Operator

The [OpenTelemetry Operator](https://opentelemetry.io/docs/platforms/kubernetes/operator/automatic/) injects Python auto-instrumentation: it sets the `OTEL_*` variables and puts a `sitecustomize` hook on `PYTHONPATH` that installs global tracer and meter providers. `OtelExtender(use_sdk_defaults=True)` and `OtelMetricsExtender(use_sdk_defaults=True)` use them with no setup code, and spawned `MULTIPROCESSING` workers inherit `PYTHONPATH`, so they need no `child_bootstrap`.

```yaml
apiVersion: opentelemetry.io/v1alpha1
kind: Instrumentation
metadata:
  name: mloda
spec:
  exporter:
    endpoint: http://otel-collector:4318  # Python auto-instrumentation exports OTLP over HTTP
  propagators: [tracecontext]
  python:
    env:
      - name: OTEL_PYTHON_DISABLED_INSTRUMENTATIONS
        value: requests,urllib3  # else OpenLineage http emits become client spans
---
# on the pod template
metadata:
  annotations:
    instrumentation.opentelemetry.io/inject-python: "true"
    # instrumentation.opentelemetry.io/otel-python-platform: "musl"  # Alpine images
```

The injected SDK shadows the image's own `opentelemetry` install, so pin compatible versions or leave the SDK out of the image.

### Pod attributes in the Collector

The Operator already sets `k8s.pod.name` and `k8s.namespace.name` in `OTEL_RESOURCE_ATTRIBUTES`. For labels and annotations, or for pods without the Operator, add the Collector's `k8sattributes` processor (its service account needs read access to pods and namespaces, plus the workload kinds you extract; see the processor's README for your Collector version). It enriches OTel signals only, never OpenLineage events.

```yaml
processors:
  k8sattributes:
    extract:
      metadata: [k8s.namespace.name, k8s.pod.name, k8s.node.name, k8s.job.name, k8s.cronjob.name]
      labels:
        - tag_name: app
          key: app.kubernetes.io/name
          from: pod
service:
  pipelines:
    traces:
      processors: [k8sattributes, batch]
```

### Jobs and CronJobs

A Job that finishes exits normally and flushes at exit. `activeDeadlineSeconds` and a CronJob's `concurrencyPolicy: Replace` end it with SIGTERM, so install the handler there too. A Collector sidecar must be a native sidecar (an `initContainer` with `restartPolicy: Always`, Kubernetes 1.29+), else the Job never completes. Each run is a new pod and process, so aggregate its metrics as in [Metrics from short-lived processes](#metrics-from-short-lived-processes): run the merge only on the central gateway (a sidecar may forward to it), and set `OTEL_SERVICE_NAME` and `OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE=delta` in the Job container's `env` (under the Operator, an Instrumentation's `python.env` would switch every workload using it).

### Serverless

Lambda freezes the process after the handler returns and Cloud Run throttles its CPU, so exit-time flushes may never run. Before returning, call `force_flush()` on the tracer and meter providers. For OpenLineage use the synchronous `http` transport, which sends each event as it is emitted; do not call `OpenLineageExtender.close()` on a warm instance you reuse, since a closed extender raises on its next use (build one per invocation if you must close it). Lambda sends SIGTERM only when an extension is registered, with about 500 ms to react, too short for the handler. Cloud Run sends SIGTERM 10s before SIGKILL, so install the handler with a smaller `grace`.

## Multiprocessing, threads and asyncio

- **`MULTIPROCESSING`:** each spawned worker needs its own provider. Add `parallelization_modes={ParallelizationMode.MULTIPROCESSING}` and `child_bootstrap=install_tracer_provider` (picklable, from an importable module; call `install_meter_provider` inside it too for metrics) to `run_all`, `stream_all`, `run` or `stream_run`, and use `OtelExtender(use_sdk_defaults=True)` and `OtelMetricsExtender(use_sdk_defaults=True)`. Worker spans join the run's trace on their own. `OpenLineageExtender(use_sdk_defaults=True)` builds its client per worker; an injected client survives only if it pickles.
- **Flush on worker exit:** core calls each extender's `close()` when a worker exits, within `graceful_shutdown_timeout` (default 2s, shared by the worker's extenders). With a `BatchSpanProcessor`, raise it together with the extender's `close_timeout` attribute (default 1s, e.g. `extender.close_timeout = 5.0`) so the batch drains. Details in [Pickle Compatibility](11-create-extender.md#pickle-compatibility).
- **Metrics in workers:** each extender's `close()` flushes its own provider within its own `close_timeout`, all sharing the worker's `graceful_shutdown_timeout`; raise it together with the close timeouts when several extenders flush. Each worker exports as its own process; to aggregate across workers see [Metrics from short-lived processes](#metrics-from-short-lived-processes).
- **Continue a trace from another process:** pass a W3C trace-context `carrier` to `run_all`, `stream_all`, `run` or `stream_run`. The run's root span becomes its child (with `trace_scope="plan"` the plan span stays the parent and the carrier becomes a span link). `inject_carrier()` from `mloda.community.extenders.otel.otel_multiprocessing` builds one from the current context without baggage. `env_carrier()` builds one from `TRACEPARENT`/`TRACESTATE` (see [Run under an orchestrator](#run-under-an-orchestrator)).
- **`THREADING`:** nothing extra; steps find their run's root span by run id.
- **asyncio:** the active span and `verified_context` are context variables. `asyncio.to_thread(...)` copies them; `loop.run_in_executor(...)` does not, so use `loop.run_in_executor(None, contextvars.copy_context().run, fn)`.
