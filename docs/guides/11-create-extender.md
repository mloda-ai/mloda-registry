# Create an Extender Plugin

Add cross-cutting concerns (logging, tracing, metrics) to mloda pipelines.

## Decision Tree

```text
Q1: What do you want to wrap?
    Feature calculation → FEATURE_GROUP_CALCULATE_FEATURE
    Input validation   → VALIDATE_INPUT_FEATURE
    Output validation  → VALIDATE_OUTPUT_FEATURE
    Feature matched    → FEATURE_GROUP_MATCHED
    Input data loads   → INPUT_DATA_LOAD
    Data joins         → JOIN

Q2: Need execution order control?
    YES → Set custom priority (lower runs first, default 100)

Q3: Need state with ParallelizationMode.MULTIPROCESSING?
    YES → Use class-level storage (pickle-safe)
```

## Required Methods

| Method | Required | Description |
|--------|----------|-------------|
| `wraps()` | Yes | Return `Set[ExtenderHook]` of hooks to wrap |
| `__call__(func, *args, **kwargs)` | Yes | Wrap and execute the function |
| `priority` | No | Execution order (lower = first, default 100) |
| `raise_on_error` | No | If `True` (default), a failure of this extender breaks the calculation. Set `False` for warning-only extenders |
| `never_fall_back` | No | For gates: `True` makes a failure always propagate, whatever `raise_on_error` says. Default `False` |
| `use_sdk_defaults` | No | For an extender with an external sink: `True` delegates sink resolution to the vendor SDK's own defaults. Default `False` keeps the extender inert until a client/provider is injected |

## Available Hooks

| Hook | When It Runs |
|------|--------------|
| `FEATURE_GROUP_CALCULATE_FEATURE` | Wraps `calculate_feature()` |
| `VALIDATE_INPUT_FEATURE` | Before calculation |
| `VALIDATE_OUTPUT_FEATURE` | After calculation |
| `FEATURE_GROUP_MATCHED` | Wraps feature group resolution |
| `INPUT_DATA_LOAD` | Wraps input data loading inside the active calculation context ([details](#hook-context-for-data-loads)) |
| `JOIN` | Wraps merging joined data |

## Example

```python
from typing import Any
from mloda.steward import Extender, ExtenderHook


class MyExtender(Extender):
    def __init__(self, raise_on_error: bool = True) -> None:
        self.raise_on_error = raise_on_error

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        # Before logic
        result = func(*args, **kwargs)
        # After logic
        return result
```

## Never Change Data

An extender never changes data on any hook: it returns exactly what the wrapped function returned ([core decision](https://github.com/mloda-ai/mloda/issues/1529)). From mloda 0.14.0 on, core discards an extender's return value once it has called the wrapped function, so a returned replacement is ignored; an extender that never calls it still returns its own value. Core does not prevent mutating the loaded data or other shared arguments in place, so the rule stays on the extender. `OtelExtender`'s `mask` receives the calculate result itself (for the `capture_content` preview): return a masked copy, never mask in place. The preview is then passed through core's `scrub_credentials`, a backstop for DSNs, presigned URLs and bearer or Authorization tokens the mask misses (not a replacement for it); values under a secret-named or Authorization mapping key are redacted whole. `capture_content` defaults to `None`: the `MLODA_OTEL_TRACE_CONTENT` env var decides, and an explicit `False` wins over it. A mask is required: `capture_content=True` without one raises `ValueError` at construction, and when the env var enables capture on an extender without a mask, no preview is recorded and one WARNING is logged per instance (per copy after pickling). Under `MULTIPROCESSING` the mask must pickle (a module-level function, not a lambda).

## Chaining and Error Handling

Multiple extenders for the same hook chain automatically (sorted by priority, lower first).

Override `on_plan_start(plan)` and `on_plan_complete(plan, outcome)` for planning (exceptions are logged, never propagated), and `on_run_start(run, plan, steps)` (parent, once per run, before setup; an exception refuses the run when `raise_on_error` or `never_fall_back` is set). Override `on_run_complete(self, run: RunContext, outcome: LifecycleOutcome)` (both from `mloda.steward`) to act once a run ends. It fires once per run in the parent, in every mode, with a succeeded, failed or cancelled outcome. An exception is logged, unless the extender sets `raise_on_run_complete = True` and the outcome succeeded, then it is re-raised after every extender was called.

`PlanContext.structure_hash` is a deterministic plan fingerprint (`mloda.steward.plan_structure_hash`), equal across runs, processes and sessions of the same plan, unlike `plan_id`. It is `None` in `on_plan_start` and set in `on_plan_complete` (after a successful plan) and `on_run_start`.

Extender failures are breaking by default, both for a single extender and in a chain: the exception propagates and the calculation fails. An extender opts into warning-only behavior by setting `raise_on_error = False` (commonly a constructor argument). Its failure is then logged as a warning and the wrapped function still runs. Non-critical or observability extenders should pass `False`.

Only the extender's own failure is caught. An exception raised by the wrapped function always propagates, and the wrapped function is never run twice.

An extender that refuses a call by raising (a gate) sets `never_fall_back = True`, so the refusal propagates whatever `raise_on_error` says; without it, a warning-only extender that raises before delegating is logged and the wrapped function runs anyway. Give a gate a `priority` strictly below every other extender on its hooks, so it runs outermost and no outer extender that catches errors from `func` can swallow the refusal; ties break deterministically: `never_fall_back` extenders first, then `priority`, then class module and qualified name. Core selects each hook's extender once per run, and `never_fall_back` gates hold even against a buggy plugin or extender. See core's [error handling](https://github.com/mloda-ai/mloda/blob/0.15.1/docs/docs/chapter1/extender.md#4-error-handling).

## Pickle Compatibility

Only needed with `ParallelizationMode.MULTIPROCESSING`. Avoid unpicklable instance variables (locks, tracers, connections). Use class-level storage or create resources lazily in `__call__()`.

Both extenders trial-pickle an injected sink (`client` for `OpenLineageExtender`, `tracer_provider` for `OtelExtender`) whenever a copy is made (worker processes under `MULTIPROCESSING`). A picklable sink is pickled as-is and survives into the worker. An unpicklable one is dropped instead: logged once per instance at WARNING, naming the sink and the underlying pickle exception's type, and the resulting copy is inert unless `use_sdk_defaults=True`. With `use_sdk_defaults=True`, `OtelExtender` resolves a provider installed in the worker via `child_bootstrap` passed to `run_all`, and `OpenLineageExtender` builds its own client lazily per worker. For your own extender with an injected sink, create a `WarnOncePerInstance` from `mloda.steward` in `__init__` and warn through it in `__getstate__` when `pickle_failure_reason` reports the sink unpicklable, as both extenders do.

A self-built `OpenLineageExtender` client is closed by `close()`, when its extender is garbage collected (on the thread that drops the last reference, capped at `close_timeout` as read when the client is built), or at interpreter exit in the parent (up to 10s per extender). `MULTIPROCESSING` workers rely on core's `close()`. A transport's close is one-shot, so a close with no budget left (`close(0)`, or a spent worker deadline) does not close an injected client: it returns `False` and leaves the flush to a later closer with budget.

Core calls each worker copy's `close()` when a `MULTIPROCESSING` worker exits, normally or after an error, best effort within the run's `graceful_shutdown_timeout` (default 2.0s, shared by that worker's extenders); the parent-death watchdog path is best effort too and can skip it (see core's `Extender.close`). Each registry extender flushes its sink in `close()`, capped at its own `close_timeout` (default 1s, on the sink for `OtelLogAuditSink`), so one slow sink cannot starve the others. `close_timeout` is a class attribute, not a constructor argument: set it on the instance (`extender.close_timeout = 5.0`) or a subclass. Inside a worker close, each flush is also capped at what is left of the shared deadline (`capped_close_timeout` in `mloda.community.extenders.shared.teardown`), even for a no-cap `close_timeout` or `OpenLineageExtender.close(-1)`. Close order is not guaranteed, so a late extender can get little or no budget; raise `graceful_shutdown_timeout` (`run_all`, `stream_all`, `Session.run`/`stream_run`) together with `close_timeout` if a sink must drain fully. OpenLineage's `async_http` transport still joins its worker thread without a limit after the capped wait, so an in-flight request can overrun both. That makes a buffered sink fine under `MULTIPROCESSING`: OpenLineage `async_http` or kafka transports, and an OTel `BatchSpanProcessor` (`BatchLogRecordProcessor` for `OtelLogAuditSink`) installed by `child_bootstrap`, as long as the flush fits inside `graceful_shutdown_timeout`. Events past the budget are still lost, though a flush that hits its `close_timeout` keeps running in the background rather than being cancelled, until the worker process exits. `AuditExtender.close()` delegates to the sink's own optional `flush()`, so only `OtelLogAuditSink` is capped this way (its own `close_timeout`); a custom sink's `flush()` has no cap unless it applies one itself, e.g. via `capped_close_timeout`. The `user_hash_key` of `OtelLogAuditSink` travels in the pickled sink under `MULTIPROCESSING`, so treat that pickle as secret material and load the key from a secret store or env var rather than hard-coding it.

## Emitting on the calculation thread

Extender code runs inline with the wrapped call: a blocking sink stalls every call, and with `raise_on_error=True` a sink failure fails the run. For OpenLineage under SYNC and THREADING, prefer the `async_http` transport or a short timeout (via `OPENLINEAGE_CONFIG` or `OPENLINEAGE__TRANSPORT__*`), and keep `raise_on_error=False` for observability. After a transport failure (connection, timeout, HTTP 5xx/408/429) in a run, `OpenLineageExtender` skips emission for that run's new steps for a minute with a WARNING per transport failure (the first failure still costs the transport's full retry budget). Steps already started still emit their terminal event, and other emit errors never trip it. Only OSError-based failures (requests transports such as http) trip it; other transports (kafka, composite, cloud SDKs) never do. The skip is per process (each `MULTIPROCESSING` worker pays the first failure itself) and per run, and resets when core reports the run complete. With `raise_on_error=True` nothing is skipped and a START failure fails that step (terminal-event failures are always logged, never raised). Under `MULTIPROCESSING`, see [Pickle Compatibility](#pickle-compatibility) for how `close()` bounds a buffered sink's flush on worker exit.

## Sink Resolution

Adding an extender opts a pipeline into instrumentation, not into ambient configuration. `use_sdk_defaults` is the explicit opt-in for that. Resolution order, strict:

1. An injected client/provider wins. No other sink is resolved alongside it.
2. Else `use_sdk_defaults=True` delegates fully to the vendor SDK's own resolution (globals, env vars, config files, and a console fallback where the SDK has one).
3. Else the extender is inert: no vendor configuration consulted, no backend constructed, nothing emitted. The wrapped call still runs and its result is returned unchanged. Logged once per instance at WARNING, including after a pickle round trip that drops an injected sink.

`mloda-community-otel` ships `opentelemetry-api` only, whose default tracer provider is a no-op with no console fallback, so `OtelExtender(use_sdk_defaults=True)` exports nothing until an SDK `TracerProvider` with an exporter is configured. Install `opentelemetry-sdk` and call `opentelemetry.trace.set_tracer_provider` (under `MULTIPROCESSING`, in each worker via `child_bootstrap`). With only the API default present it logs one WARNING per instance (per copy after pickling); to run without an SDK on purpose, inject `opentelemetry.trace.NoOpTracerProvider()` as `tracer_provider`. Any extender whose `use_sdk_defaults` can resolve to a vendor no-op should warn once the same way.

For OpenLineage, `use_sdk_defaults=True` honours these ambient sources in precedence order: `OPENLINEAGE_DISABLED`; config from `OPENLINEAGE_CONFIG`, `./openlineage.yml` or `~/.openlineage/openlineage.yml` (merged with `OPENLINEAGE__*` env vars); `OPENLINEAGE_URL` (with `OPENLINEAGE_API_KEY`, `OPENLINEAGE_ENDPOINT`); then the console fallback, which logs full events at INFO and a WARNING per built client.

## Usage

```python
from mloda.user import mloda

results = mloda.run_all(features=["my_feature"], function_extender={MyExtender(), OtherExtender()})
```

Extenders are session-level: pass them to `run_all` or when preparing the session. `Session.run()` takes none.

## Reading the hook context

`HookContext.current()` inside `__call__` returns the context of the hook being dispatched, or `None` outside a hook call (a direct call, a unit test without an activated context). It is set only on the dispatching thread, so capture it before handing work to another thread, or run that work under `contextvars.copy_context().run`.

- Before `func` runs, on the calculate and validate hooks: `hook`, `feature_group_class` (`module.qualname`), `feature_group_version`, `plugin_version`, `feature_names`, `specialized_from`, `input_features`, `input_feature_edges`, `compute_framework_name`, `rows_in`, `run_id`, `plan_id`, `step_uuid` (the step's `PlanStep.step_uuid`, also on `INPUT_DATA_LOAD`; stable across a session's runs), `carrier`, `worker_index`, and the verified `tenant_id`, `project_id` and `principal` (see [Verified run context](#verified-run-context)). `inject_carrier` and `extract_carrier` carry the W3C traceparent only, never baggage.
- After `func`: `duration_seconds` and `status` (`"success"` or `"error"`, for the wrapped call only), also when it raises; on success, `rows_out` (calculate, data loads) and `output_schema` (calculate, validate-output).
- Per hook: `INPUT_DATA_LOAD` inherits the calculation's identity fields and adds the data-access fields (see [Hook context for data loads](#hook-context-for-data-loads)). `JOIN` and `FEATURE_GROUP_MATCHED` carry no feature-group identity, only `run_id` (`plan_id` on the match hook), the verified identity and their own fields: `join_type`, `join_keys` and the link's declared `join_left_feature_group` / `join_right_feature_group` (`module.qualname`), or `feature_names` and the running `plan_feature_count`, `plan_node_count` and `plan_depth`; the match hook sets `feature_group_class` only after `func` returns. `feature_group_class`, `feature_group_version` and `compute_framework_name` are `str | None`: `None` on `JOIN` (class and version; it keeps `compute_framework_name`) and on the match hook (class until resolution, version and framework always), so a match-time audit refusal records all three as `null`. `FEATURE_GROUP_MATCHED` runs with `run_id=None` and `plan_id` set. Other fields: `declared_attributes`, `asof_config` (on `JOIN`), and `reader_class` and `data_access_identity_is_fallback` (on `INPUT_DATA_LOAD`). Unfilled fields keep their defaults (`None`, `""` or `()`).

```python
import logging
from typing import Any

from mloda.steward import Extender, ExtenderHook, HookContext

logger = logging.getLogger(__name__)


class FactsExtender(Extender):
    def __init__(self) -> None:
        self.raise_on_error = False

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        finally:
            context = HookContext.current()
            if context is not None:
                logger.info(
                    "%s took %ss, status %s", context.feature_group_class, context.duration_seconds, context.status
                )


# Outside a run, activate a hand-built context (tests can use make_hook_context from
# mloda.testing.extenders.hook_context). Core fills duration_seconds and status only
# during a run, so both log as None here.
logging.basicConfig(level=logging.INFO)
context = HookContext(
    hook=ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE,
    feature_group_class="my_plugin.MyFeatureGroup",
    feature_group_version="1",
    compute_framework_name="PyArrowTable",
)
with context.activate():
    FactsExtender()(lambda: "result")
```

See core's [HookContext facts](https://github.com/mloda-ai/mloda/blob/0.15.1/docs/docs/chapter1/extender.md#5-reading-call-facts-via-hookcontext) for the full field semantics.

## Hook context for data loads

`INPUT_DATA_LOAD` runs nested inside the active `FEATURE_GROUP_CALCULATE_FEATURE` call, but an extender may wrap it alone: the calculation context is activated whenever either hook has an extender registered, so a data-load wrapper still fires when nothing wraps `FEATURE_GROUP_CALCULATE_FEATURE`.

Its `HookContext` inherits the enclosing calculation's feature-group and feature identity fields (`feature_group_class`, `feature_group_version`, `plugin_version`, `feature_names`, `input_features`, `input_feature_edges`), then adds `data_access_identity` and `data_access_format` for the input being read. The enclosing calculation context does not carry those data-access fields, and `rows_in` is left unset here.

The `data_access_identity` that core supplies on the context comes from the reader's default-deny `data_access_identity(data_access)` classmethod (see `BaseInputData.data_access_identity`); a reader may override it to publish more, and then owns what it publishes.

Record `context.data_access_identity` as given, never the raw data access (`args[0]` of the wrapped load call), which can carry credentials that core's identity leaves out. `AuditExtender`, `OpenLineageExtender`, `LineageFacetsExtender` and `OtelExtender` all do: it is the audit entry, the OpenLineage `dataSource` name, the source of the dataset name (and its dedupe key), and the `mloda.data_access.identity` span attribute. With core's default, a keyword DSN or ODBC string is recorded as `str`, a mapping as its sorted key names (`{host, password}`), and a URI as its scheme, host and path, without user information, query or fragment. Azure `abfs`, `abfss`, `wasb` and `wasbs` URIs keep the container, so `abfss://raw@acct...` and `abfss://curated@acct...` with the same path stay two datasets. Known limits: sources core does not tell apart become one dataset, namely HTTP sources that differ only in their query, `jdbc:` URIs (cut to the host), percent-encoded paths (cut to the directory), every string core cannot parse or find (a DSN, a missing path, a URI with a space, several hosts or an `@` in its query), which are all `str`, and any other data access (a number, a missing `PurePath`, a connection object), recorded as its type name. A reader that needs a finer identity overrides `data_access_identity` (see [Non-File / HTTP Sources](feature-group-patterns/27-input-data-readers.md#non-file--http-sources)). Identities recorded by earlier registry versions, which sanitized the raw data access, do not join to these. `data_access_identity`, `data_access_format` and `data_access_identity_is_fallback` are index-aligned lists with one entry per distinct (identity, format, flag) triple the call attempted, in first-seen order: repeats collapse, a load without an identity is omitted, and `[]` means none. The flag is core's `data_access_identity_is_fallback`, recorded as given; True means the identity is a placeholder (a type name or mapping key names), not a dataset id. `OtelExtender` sets it on the load span as `mloda.data_access.identity_is_fallback` (`true` or `false`, omitted when unset). `OpenLineageExtender` and `LineageFacetsExtender` mark an input dataset with an `mlodaDataAccess` facet (`identityIsFallback: true`) when any fallback load of that name occurs, instead of dropping it. An absent key means not recorded, since keys may be added within a `record_version`.

Feature names are published as given wherever they become dataset names: declared input features and outputs in `OpenLineageExtender`, and also column-lineage edges and validation datasets in `LineageFacetsExtender`. A consumer's input name then equals its producer's output name, which is what joins the lineage graph. A feature name is an author- or caller-chosen identifier, so never put a secret in one. OpenLineage load datasets follow the OpenLineage naming spec: `s3://bucket/key` and `gs://bucket/key` become namespace `s3://bucket` or `gs://bucket` with name `key`; `abfss://container@account.dfs.core.windows.net/path` becomes that namespace with name `path`; `file:///abs/path` and an existing absolute local path become namespace `file`. Every other identity (`jdbc:`, `http(s)`, `hdfs`, `s3a`, `wasbs`, a relative or missing path, a type name, mapping key names, or one with a query, fragment or `@` outside the abfss container) keeps `dataset_namespace` and the identity as its name, so `dataset_namespace` no longer applies to mapped loads. The `dataSource` facet name stays the identity, the join key to audit records and the `mloda.data_access.identity` span attribute. A feature named like a load's data access is merged with that load's dataset only for an unmapped identity whose name equals the feature name; a mapped load dataset no longer merges with such a feature.

## Declared attributes

A `FeatureGroup` or reader declares span attributes with a `declared_attributes(features)` classmethod. Core calls it (the feature group's on calculate, the reader's on load) and hands the scalar result to extenders as `HookContext.declared_attributes`; a raise or a non-mapping return degrades to `None` with a core WARNING. `OtelExtender` sets each entry as `mloda.declared.<key>` before the wrapped call runs, on calculate and load spans only. Metadata only: do not echo option values back, since `FeatureSet` options can carry credentials.

- Skipped when the span is not recording, or when nothing is declared.
- Only scalar values (`str`, `bool`, `int`, `float`) are kept; `str` is truncated to the content preview cap.
- At most 32 declared keys are set (first 32 in mapping order), so core attributes are never evicted by the SDK's limit.

## Verified run context

Use `mloda.steward.verified_context()` around the run call when an extender needs server-verified tenant, project, or principal identity. These values populate `tenant_id`, `project_id`, and `principal` on every `HookContext` created for that run and cannot be overridden through feature `Options`.

`FEATURE_GROUP_MATCHED` reads the scope active at plan time (`prepare`, `explain`, `diagnose`, the planning half of `run_all`); every other hook reads the scope active at run-call time. For a session prepared under one scope and run under another, core refuses (`GateBypassError`) a run whose identity (`tenant_id`, `project_id`, `principal`) differs from the plan identity when a `never_fall_back` match-hook gate does not override `on_run_start`; a gate that does override it must check the run identity there (see core's [HookContext facts](https://github.com/mloda-ai/mloda/blob/0.15.1/docs/docs/chapter1/extender.md#5-reading-call-facts-via-hookcontext)).

```python
from mloda.steward import verified_context
from mloda.user import mloda

with verified_context(tenant_id="tenant-42", project_id="project-7", principal="service-account"):
    results = mloda.run_all(features=["my_feature"], function_extender={MyExtender()})
```

`AuditExtender(sink, fail_closed=True)` (`mloda-enterprise-audit`) turns a missing required identity into a refusal: it writes the deny record, then raises `IdentityRequiredError` before the wrapped call runs, so no feature is calculated. It sets `priority = 0` so the gate runs outermost; an extender at priority 0 or lower may still run outside it. It gates two hooks:

- `FEATURE_GROUP_MATCHED` reads `verified_context` at plan time, so `prepare`, `explain` and `diagnose` need the scope too, not only `run`. It writes a record (which `OtelLogAuditSink` also emits) only when it refuses, before any feature is calculated. Without `fail_closed`, a missing identity denies at calculate and the run proceeds. No allow event is written at match; the calculate record is the allow event.
- `FEATURE_GROUP_CALCULATE_FEATURE` reads it at `run()` or `stream_run()` time. A run outside any `verified_context` inherits the plan-time identity, so a session prepared inside the scope still passes.

The refusal record has `decision="deny"`, `status="error"` and an `error_type` naming `IdentityRequiredError`; a match-time refusal records the feature-group fields as `null`. With `fail_closed=True` it also implements `on_run_start`: a run whose identity lacks a required field is refused with `IdentityRequiredError` and a `"RUN_START"` deny record, while a complete identity that differs from the plan's (prepare once, run per tenant) is allowed and audited under the run's identity. `feature_group_class`, `feature_group_version` and `compute_framework_name` are null on hooks where core leaves them unset (e.g. the match hook). It is only as durable as the sink, so use a synchronous one such as `NdjsonAuditSink` under `MULTIPROCESSING` (see [Pickle Compatibility](#pickle-compatibility)). `fail_closed=True` declares core's `never_fall_back`, so `raise_on_error` has no effect on it, also when set after construction: the refusal and a sink failure on the refusal path or after a successful call always propagate. The constructor rejects `fail_closed=True` with an empty `required_identity`. `fail_closed` is read-only, since `priority`, `never_fall_back` and the default `policy_version` are derived from it at construction. One path still fails open: a call made outside a core run (no hook context). Under registry strict mode `strict` (`PluginCollector.set_strict_mode(...)`, else `MLODA_PLUGIN_REGISTRY_STRICT`), core refuses an unregistered `fail_closed=True` extender with `EnvironmentPreconditionError` when the session is built, before any hook runs (so no audit record is written), instead of dropping it; under `warn` it still runs.

Every record carries `policy_version`: the argument of that name (a non-blank string), else a 12 hex character fingerprint of the constructor-supplied gate (sorted `required_identity` plus `fail_closed`), which does not track code changes. The key is additive within a `record_version`.

Audit records are `record_version` 2. Beyond the identity fields they carry:

- `decision` is the policy outcome (`allow` or `deny`); `enforced` is `true` only on a `fail_closed` refusal record (match, `RUN_START` or calculate), so a deny that merely records a missing identity is told apart from a refusal that stopped the run.
- `compliant` is true only when no required identity is missing and the principal is non-blank. A default config without a `principal` in `required_identity` therefore records `compliant: false` whenever no principal is in scope.
- `phase` is `"plan"` for the match hook and `"run"` otherwise.
- `input_feature_edges` (`{feature: sorted input names}`, or null) next to `input_features`.
- `host` (the process hostname), `worker_index` and `start_time` (UTC ISO, taken before the wrapped call; equal to `event_time` on refusal and `RUN_START` records).
- `step_run_id`, the deterministic id of the step (see [Run correlation](#run-correlation)); null without a `run_id` and on match and `RUN_START` records.
- `structure_hash`, the run's `PlanContext.structure_hash` (cached from `on_run_start`); null when no `on_run_start` cached one for the run (match records, hook-only use).
- `trace_id` and `span_id` (hex), best effort: the active OTel span when the audit call started, else the carrier's trace id with a null `span_id`, else null. They need `opentelemetry-api` and are null without it.

Verification does not depend on the record version, so a v2 log seals and verifies like any other.

`AuditExtender.close()` calls the sink's optional `flush()` on graceful `MULTIPROCESSING` worker exit (see [Pickle Compatibility](#pickle-compatibility)): `OtelLogAuditSink` flushes its logger provider, `TeeAuditSink` fans out to each child's own `flush()`, and a write-only sink such as `NdjsonAuditSink` needs none and is left alone.

`TeeAuditSink(*sinks)` writes each record to every sink in order; the first failure propagates and later sinks are skipped, so put the durable `NdjsonAuditSink` first. It raises `ValueError` for no sinks or a sink without a callable `write`.

`OtelLogAuditSink()` (or `OtelLogAuditSink(user_hash_key=b"...")`, which adds `user.hash`, see below) emits one OpenTelemetry log record per audit record, so allow and deny decisions reach a Collector. It needs the extra `mloda-enterprise[otel]` (`opentelemetry-api>=1.37,<2`); the package loads without it and only constructing the sink raises `ImportError` naming the extra. Allow is INFO, deny is WARN, the body is the decision and the timestamp is `event_time`. Attributes: `mloda.audit.decision`, `mloda.audit.deny_reason`, `mloda.audit.policy_version`, `mloda.audit.hook`, `mloda.audit.phase`, `mloda.audit.enforced` (bool), `mloda.step.run_id`, `mloda.run.id`, `mloda.plan.id`, `mloda.plan.structure_hash`, `mloda.tenant.id`, `mloda.project.id`, `user.hash` (only with a key), `mloda.feature_group.name`, `mloda.feature.names`, `error.type`; blank or absent values are omitted. When the record has both `trace_id` and `span_id`, the log record carries that trace context, so it correlates with the `OtelExtender` spans.

`user.hash` is omitted unless `OtelLogAuditSink(user_hash_key=b"...")` (bytes, at least 32) is given, because an unkeyed hash of a low-entropy principal (email, username) can be confirmed offline by dictionary. With a key it is `hmac.new(key, json.dumps([tenant_id, principal], separators=(",", ":"), ensure_ascii=True).encode("utf-8"), hashlib.sha256).hexdigest()` (tenant_id is None when blank; the key must be at least 32 bytes, shorter raises `ValueError`): pseudonymous, not anonymous, and the same principal under two tenants hashes differently. Use one key per deployment and a key of its own, never the manifest signing key. Join by computing the same HMAC over the NDJSON record's `tenant_id` and `principal`; the NDJSON record stays the identifying copy. Rotating the key changes every `user.hash`, so older records no longer join. `mloda.run.id` joins to `OtelExtender` spans. Data-access identities, feature values and exception messages are never included. Emitting is best effort: a failure is logged at WARNING and never fails the run. The sink looks up the global logger provider on every write and logs one WARNING per process while only the API default provider is installed. Under `MULTIPROCESSING` a spawned worker has no SDK logger provider unless `child_bootstrap` installs one (see [Pickle Compatibility](#pickle-compatibility)), and a synchronous exporter blocks the calculation thread. The OTel copy is not sealed or tamper-evident, so keep the NDJSON log and its manifest as the retained record and compose them:

```python
import os

from mloda.enterprise.extenders.audit import AuditExtender, NdjsonAuditSink, OtelLogAuditSink, TeeAuditSink

key = os.environ["MLODA_AUDIT_USER_HASH_KEY"].encode()
sink = TeeAuditSink(NdjsonAuditSink("audit.ndjson"), OtelLogAuditSink(user_hash_key=key))
extender = AuditExtender(sink, fail_closed=True)
```

### Sealing and anchoring

- `head_anchor` (a `HeadAnchor`, e.g. `NdjsonHeadAnchor`) receives each new manifest log head, and `anchored_heads` makes verification require every anchored head to be a line of the log. This detects truncation, rollback, deletion and substitution.
- The anchor and the logs need append-only or WORM storage that the log writer cannot rewrite. Key custody stays a platform duty.
- Only an anchor detects deletion or a restarted log. `log_id` only tells logs apart: a deleted log restarts with a fresh genesis and verifies.
- `log_id` needs a fresh manifest log: an existing log without a genesis fails every seal.
- A write-only anchor (like the OTel example, `latest()` returns None) cannot stop an auto-seal from chaining onto a rolled-back log. Verify offline with `verify_ndjson_log(..., anchored_heads=<heads exported to the backend>)`.
- After a quarantine repair that cut the log, write the repaired head to the anchor (`anchor.write(expected_head)`) so the latest anchored head is in the log again.
- The quarantine functions take their own `head_anchor` for the trace head (separate from the manifest log's). `verify_quarantine_log(..., anchored_heads=<every trace head anchored>)` detects a truncated trace. Pass every anchored head, not only the latest, here and to the quarantine functions, so a repair refuses to chain onto a cut trace.
- Rotation needs the outgoing key's co-signature, so a lost current key means starting a new log.
- `seal_failure_policy` is `"log"` (default), `"raise"` or a callable `(run_id, exc)`; failures (rotation failures too) are counted in `seal_failures`. Under `"raise"`, `AuditExtender` sets `raise_on_run_complete`, so a seal failure fails a run that otherwise succeeded; with `"log"` or a callable core contains it. Under registry strict mode `strict` (`PluginCollector.set_strict_mode(...)`, else `MLODA_PLUGIN_REGISTRY_STRICT`), core treats an unregistered extender with `raise_on_run_complete` (so `seal_failure_policy="raise"`) like a `never_fall_back` gate and refuses it with `EnvironmentPreconditionError` instead of dropping it.
- `seal_ndjson_runs(..., older_than=timedelta(...))` is a manual sweep for crashed runs, which stay unsealed until an operator sweeps. It seals only runs older than the threshold, marks them `sealed_late`, and skips runs without a parseable `event_time`.
- Cost: without `seal_index_path` every auto-seal verifies the whole manifest log and parses the whole audit file. With it, a seal costs the bytes from the run's first record to the end of the audit file plus the lines of runs not yet sealed (crashed runs, and runs refused at plan time, stay in that set until swept).
- The seal index is a rebuildable cache on writable storage, not the WORM storage. A missing, stale or other-key index triggers a full scan; index errors are logged, never fatal.
- Trade-off: a seal no longer re-verifies lines before the checkpoint, so run `verify_ndjson_log(..., anchored_heads=...)` on a schedule. An anchor lagging behind the checkpoint falls back to full verification.
- Negative lookups trust the index's unsigned hint rows: whoever can write the index can make a sealed run look unsealed and cause a duplicate seal. Keep the index under the same write protection as the logs' writer and verify offline on a schedule.
- `seal_index_path` requires the sealing config and must not alias the audit, manifest or anchor paths.
- A line longer than 64 MiB fails verification and is refused on write; logs written by earlier releases with a seal line over 64 MiB no longer verify.
- Re-running a sealed run and manual sweeps stay full scans (retained archives are scanned too).
- Rotate with `rotate_ndjson_segment(audit_path, manifest_path, signer=..., log_id=..., head_anchor=...)` on your own schedule; the `log_id` stays and the log needs a genesis. Pending and crashed runs are carried over, so sweep crashed runs first. Run it as the account that writes the logs (the new live files are created by the rotating process). Rotation and sealing check only the anchors written since the live segment's genesis (plus the predecessor head); pass the full anchor history to `verify_ndjson_segments`.
- `segment_max_bytes=` / `segment_max_age=timedelta(...)` make an auto-sealing `AuditExtender` rotate after a seal once the sealed bytes a rotation would archive reach that size (carried pending runs do not count) or the live segment reaches that age (needs `log_id`). Both are re-checked under the manifest lock, so concurrent sealers rotate once. A completion that finds an interrupted rotation finishes it (logged at WARNING) and retries its seal. The rotating run pays for verifying the whole outgoing segment; carried pending runs alone over the size make each completion re-read the audit file under the lock, so sweep crashed runs. A failure counts in `seal_failures` and follows `seal_failure_policy` (a callable policy then receives the rotation's own exception, the run already sealed).
- Archives are `<path>.<NNNNNN>` pairs. Deleting the oldest pairs is detected only by anchors; runs sealed in deleted or moved archives stay refused by a targeted `seal_ndjson_runs` only until the next rotation rebuilds a seal index.
- `verify_ndjson_segments(...)` verifies the retained history; `verify_ndjson_log` on one pair verifies a single segment. Online verification holds the manifest shared lock, so auto-seals wait for it.
- An interrupted rotation blocks sealing until `rotate_ndjson_segment` runs again, unless auto-rotation is on. Safe only for `NdjsonAuditSink` writers (shared flock, needs fcntl). A hard crash can leave `.<name>.*` temp files beside the logs; delete them once no rotation runs.
- Manifest logs written by earlier releases (version 1 lines) still verify; new lines appended to them are version 2.

```python
import os

from mloda.enterprise.extenders.audit import AuditExtender, HmacSha256Signer, NdjsonAuditSink, NdjsonHeadAnchor


class OtelHeadAnchor:
    """Emits each head as an OTel log record. Write-only: it emits but cannot be read back."""

    def write(self, head: str) -> None:
        from opentelemetry._logs import LogRecord, get_logger_provider

        get_logger_provider().get_logger("mloda_enterprise_audit_anchor").emit(LogRecord(body=head))

    def latest(self) -> str | None:
        return None


extender = AuditExtender(
    NdjsonAuditSink("audit.ndjson"),
    audit_path="audit.ndjson",
    manifest_path="manifests.ndjson",
    signer=HmacSha256Signer(os.environ["MLODA_AUDIT_MANIFEST_KEY"].encode(), "key-1"),
    log_id="prod-audit",
    head_anchor=NdjsonHeadAnchor("/mnt/worm/anchor.ndjson"),  # separate, append-only storage
    seal_index_path="seal-index.sqlite3",  # optional SQLite cache on writable storage, not the WORM storage
    segment_max_bytes=256 * 1024 * 1024,  # optional: rotate after a seal once the audit file reaches 256 MiB
)
```

## Tracing and run correlation

To export these spans and events to a backend (Collector, sampling, privacy defaults, multiprocessing), see [Export Telemetry to Any Backend](12-export-telemetry.md).

### OTel root span and trace_scope

`OtelExtender` starts a root span `mloda.run` in `on_run_start` (attributes `mloda.run.id`, `mloda.plan.id`, `mloda.plan.structure_hash`) and ends it in `on_run_complete` with `mloda.run.status` (`succeeded`, `failed` or `cancelled`; `failed` also sets `error.type` and span status `ERROR`). Step spans (`calculate <FeatureGroup>`, the short class name; plain `calculate` without one) are children of that root; calculate, validate and load spans carry `mloda.step.uuid` (core's `step_uuid`) when set. Join spans (`join <join_type>`, plain `join` without a type) are also children of the root and carry `mloda.join.type`, `mloda.join.left_feature_group` and `mloda.join.right_feature_group` (`module.qualname`), `mloda.join.keys` (`left=right` column pairs, so column names are exported, never values) and, for as-of joins, `mloda.join.asof.*` (time columns, `direction`, `allow_exact_matches`, and `tolerance`, a number in the time column's units, or `tolerance_seconds` for a `timedelta`); unset ones are omitted. The root's parent is the run carrier if there is one, else the caller's active span.

`trace_scope` picks the trace shape; anything else raises `ValueError`:

- `"run"` (default): no plan span, one trace per run. A failed plan still gets an `ERROR` `mloda.plan` span with `error.type`, under the caller's active span, else in its own trace; that includes a failing `explain` (which raises) and `diagnose` (which returns a diagnosis).
- `"plan"`: `on_plan_start` opens a `mloda.plan` span that parents every run of the plan and gets `mloda.plan.structure_hash` when it ends; the caller's active span or the carrier becomes a span link instead of the parent. If the plan span is unknown, the run is parented as in `"run"` mode.

Caveats of `"plan"` mode: a span has one parent, hence the link; head sampling keeps or drops all runs of a plan together; a long-lived plan (prepared once, run often) makes one long trace. `explain` and `diagnose` also emit plan spans that no run follows.

A sampled-out root drops its children. Once a root exists, the trace id is the SDK's or the caller's; the `run_id`-derived trace id is only the fallback for hook-only use, without a root. A root is stored only when the started span's context is valid; with a no-op or inert provider that is the carrier's or the caller's span, if any, so worker spans then hang directly under it. Plan mode keeps the span contexts of the most recently used 1024 plans; a run refreshes its plan, and runs of an evicted plan fall back to run-mode parenting.

### Selecting OTel spans

Step span names are display names and can change in any release. Select step spans by attribute: `mloda.operation.name` (`calculate`, `load`, `validate` or `join`) is set on every hook span with a hook context; narrow with `mloda.run.id`, with `mloda.feature_group.name` (`module.qualname`, not the short name in the span name; join spans carry `mloda.join.left_feature_group` and `mloda.join.right_feature_group` instead) or, on calculate spans, `mloda.step.run_id`. Root spans carry no `mloda.operation.name`; select them by their fixed names `mloda.run` and `mloda.plan`. The names `mloda.validate.input` and `mloda.validate.output` are also fixed and are the only way to tell the two validate spans apart. The `mloda.*` attribute keys and the `mloda.operation.name` values are stable; a change to them or to a span name is named in the release notes.

### Run correlation

`OpenLineageExtender` (and `LineageFacetsExtender`) emit a parent run in `on_run_start` and `on_run_complete`: START, then COMPLETE, FAIL (failed) or ABORT (cancelled) for the job `root_job_name`, with `runId` equal to the mloda `run_id` and the run facet `mlodaPlan` (`planId` and `structureHash`, the plan fingerprint; not the per-step `structureHash` of the `mloda` facet below). Step runs reference it through the `parent` facet and carry no `mlodaPlan`. A parent START transport failure with `raise_on_error=False` trips the run's breaker, and with `raise_on_error=True` it refuses the run; no parent events are emitted when the extender is inert, and no terminal event when the START was not emitted.

Step run ids derive from the run: `step_run_id(run_id, job_name, feature_names, compute_framework_name, step_uuid)` (`mloda.community.extenders.shared`) is a UUIDv5 over the run id and `[job, sorted feature names, framework, step uuid]`, or `None` when the run id is missing or not a UUID. OpenLineage uses it as the step `runId`, `AuditExtender` as the record's `step_run_id`, and `OtelExtender` as `mloda.step.run_id` on calculate spans, so the three join on it; all pass core's `HookContext.step_uuid`, so two steps of one run never share an id. Without a step uuid (hook-only use) the key omits it, and such steps can collide. Ids only join between extenders of the same registry release.

## Lineage facets

`LineageFacetsExtender` (`mloda-enterprise-lineage`) is used instead of `OpenLineageExtender`, not next to it. It needs the extra `mloda-enterprise[openlineage]`; without it, or with a community OpenLineage of another major.minor, the extender is not registered. Beyond the community events it adds:

- `columnLineage` on each output dataset: DIRECT edges from the step's declared input features, or, for a root step that declares its source column, from the one dataset it loaded.
- `dataQualityAssertions` on validation runs: one assertion named after the validator, `success` true or false.
- a `mloda` run facet: `featureGroupVersion`, `pluginVersion`, `computeFramework`, `declaredMasking` and `structureHash`.

`declaredMasking` is the sorted names of the features that declare masking. Masking is declared, never inferred: set the class attribute `masking = True` on the feature group, or the option `masking=True` in the feature's own `context`. It must be the boolean `True`; a `group` key, the string `"true"` and a context key the step only received from another step do not count. It is unrelated to core's `mask`. The extender emits no standard OpenLineage `masking` transformation, since nothing enforces masking yet; the declaration is only recorded in the `mloda` facet and `structureHash`.

Each output's column-lineage edges come from its own declared inputs, via core's `HookContext.input_feature_edges`. An output with no entry there (such as an injected filter or index feature) falls back to every step input (an over-approximation), with the transformation `description` `step-level declared inputs` on a multi-output step.

A root step has an edge only when it declares the source column (a feature name equalling a column is common but not guaranteed). Declare it with the option `lineage_source_column` in the feature's own `context` (a column name, or `True` for the feature name) or the class attribute `lineage_source_column` (a dict from feature name to column, or `True` for every feature name); a valid option wins. The edge runs from the loaded dataset, named as its input dataset is: the namespace and name derived from the recorded `data_access_identity` (see [Hook context for data loads](#hook-context-for-data-loads)); distinct sources with equal identities, such as dict credentials with the same keys or two keyword DSNs (both `str`), count as one dataset (any fallback load of a name marks its dataset with `mlodaDataAccess`). The reader is described once per dataset, synchronously on the calculation thread, and only when a single edge is possible (`describe_columns`, see [Column Discovery](feature-group-patterns/27-input-data-readers.md#column-discovery)). The declared column is matched exactly (case-sensitive) against that result: a match gives an edge, and a result naming other columns gives no edge and logs a WARNING. A result that is not a mapping of column names, or a describer that raises, counts as undescribable and keeps the edge unverified. Loads of one dataset merge their described columns, so one load that cannot describe leaves the dataset unverified. A `group` key, a key the step only received, `False`, an empty string and any other type do not count, and the class attribute then still applies. A root step that loads no dataset (`DataCreator`) or several distinct ones has no edge, and a step with declared inputs keeps only its input edges.

Validation is reported only for a feature group that overrides `validate_input_features` or `validate_output_features`. Each overridden validator is its own run per step, a sibling of the calculate run and both under the root run (job `<feature group>.<method>`, START then COMPLETE, FAIL or ABORT), so it adds one extra run and two events per step. The terminal event carries one input dataset per validated feature. Validate-input is skipped when the step has no data yet (root steps); when data exists it runs and the validated datasets fall back to the feature names if no inputs are declared. Core's own `EmptyResultError` and `DataTypeValidator` failures happen outside the hook, so they produce no assertion.

A failing validator is re-raised and marks every validated dataset `success=false`, because the failing one is unknown. Its message reaches neither the WARNING log (only the exception type does) nor an event.

Validation runs carry the parent facet only, not the `mloda` facet. Tie one to a calculate run through the job name and the run id.

Validate-output assertions ride on input datasets of the `<feature group>.validate_output_features` job, the only spec-legal home for `dataQualityAssertions`.

`structureHash` is the sha256 of the feature group class, versions, compute framework, feature names, declared inputs, masked features and, for a root step, the declared source column of each feature (option or class attribute). It covers the declaration, not whether an edge was emitted. It holds no run id or time, so it is stable across runs. It also changes with the feature group source and the mloda version, because both are part of `feature_group_version`.

An option (`masking`, `lineage_source_column`) counts only when it is the step's own: declared in the feature's `context` before mloda starts resolving it (core's `Options.own_context_keys`). A key the step only received never counts, whether forwarded (`propagate_context_keys`, `inherit_context_keys`) or written by a feature group while matching. A `PROPERTY_MAPPING` default is filled in by core, so it never counts either; use the class attribute for a class-wide declaration. A key the step declares itself counts even when its consumer holds or forwards the same value. A consumer that hands its `Options` (or a copy) to an input feature declares its own keys on that input too, so build a new `Options` for inputs to keep them apart. When several consumers request the same feature, a key is own if any request declared it, so `declaredMasking` and `structureHash` do not depend on which consumer core kept.

The `mloda` facet's schema URL points at its module in this repository; it is not a hosted JSON schema.

## Testing

`mloda-testing` ships three test mixins plus helpers (`make_hook_context`, `run_value_int`, `failing_feature_group`) so every extender's test suite exercises the same shared behavior:

- `ExtenderContractTestMixin` (`mloda.testing.extenders.contract`), for every extender
- `OtelExtenderTestMixin` (`mloda.testing.extenders.otel`, install `mloda-testing[otel]`), for extenders that emit OTel spans
- `OpenLineageExtenderTestMixin` (`mloda.testing.extenders.openlineage`, install `mloda-testing[openlineage]`), for extenders that emit OpenLineage RunEvents

The OTel and OpenLineage mixins both enforce the same observability mandate: a wrapped failure is logged at WARNING with the extender name and exception type, never the message, and the message never reaches a span or an event.

### ExtenderContractTestMixin

Required host hooks: `extender_class`, `make_extender`, `own_failure`. Optional: `raise_on_error_default`, `expected_hooks`, `pickled_copy_environment`, `supports_warning_only` (return `False` for a host with no warning-only mode, such as a gate declaring `never_fall_back`; the `run_all` wrapped-failure test then uses the default `raise_on_error`), `context_identity` (a dict of `tenant_id` / `project_id` / `principal` for an identity-gated extender; carried by every contract context and `run_all`; defaults to empty), `has_backend_sink` (return `True` for an extender with an external sink, and override `ambient_sink_environment` plus `sink_resolution_spy`, and optionally `ambient_sink_captured(spy)`, which returns what the ambient sink(s) the spy handed out captured once the call is done; its base default `None` opts out of the emission check, for a host that cannot expose a probe; `make_unconfigured_extender`/`make_sdk_defaults_extender` default to `extender_class()()` and `extender_class()(use_sdk_defaults=True)`), `supports_pickled_sink_capture` plus `injected_sink_capture` (`True` when a picklable injected sink survives pickling; defaults to `False`), `supports_unpicklable_sink_degrade` plus `make_unpicklable_sink_extender` (`True` when the host trial-pickles its sink and drops+warns instead of hard-failing pickling; defaults to `False`), `sink_noun` (the word the drop-warning uses for the sink, e.g. `"tracer_provider"`; `None` skips noun-specific assertions), `unpicklable_sink_failure_type` (the pickle-failure exception type name the drop-warning names; defaults to `"TypeError"`, correct for a sink holding a `threading.Lock`).

A host with `has_backend_sink() == True` must also implement `make_extender_with_sink_probe` (an extender wired to a fresh, per-instance in-memory sink, plus a zero-arg callable returning what that exact sink captured; must be per-instance state, never shared/class-level, or the identity test below is vacuous). Real-worker `MULTIPROCESSING` coverage is opt-in, separate from `has_backend_sink`: override `supports_real_worker_sink` to `True` and implement `make_real_worker_extender_and_marker(tmp_path)` (an extender wired to a file-backed sink under `tmp_path`, plus the marker file a spawned worker's emission writes to) only on a host that actually wants this coverage; it spawns a real subprocess, so it is deliberately not inherited automatically the way `has_backend_sink` is, and a lightweight self-test host should leave it at the default `False`.

The real-worker test needs `ParallelRunnerFlightServer`, which has no public re-export and cannot be imported anywhere under `mloda/` (this repo's internal-import guard forbids it). Its `flight_server` fixture instead lives in a `conftest.py` at the repo root, outside the guard's scan, and the contract test reaches it lazily via `request.getfixturevalue("flight_server")` rather than a parameter or an import. A downstream host outside this repo that opts into `supports_real_worker_sink()` must supply its own `flight_server` fixture: this repo's lives in its root `conftest.py`, but the published `mloda-testing` package does not ship one.

```python
from contextlib import AbstractContextManager
from typing import Any
from unittest.mock import patch

from mloda.testing.extenders.contract import ExtenderContractTestMixin


class TestMyExtenderContract(ExtenderContractTestMixin):
    @classmethod
    def extender_class(cls) -> type[MyExtender]:
        return MyExtender

    def make_extender(self, *, raise_on_error: bool | None = None) -> MyExtender:
        if raise_on_error is None:
            return MyExtender()
        return MyExtender(raise_on_error=raise_on_error)

    def own_failure(self) -> AbstractContextManager[Any]:
        return patch.object(MyExtender, "__call__", side_effect=RuntimeError("boom"))
```

Observability extenders that default to warning-only override `raise_on_error_default()` to return False.

`extender_class` names the class under test. `make_extender` returns an instance wired to an in-memory backend, never a real network sink. `own_failure` makes the extender's own code fail (not the wrapped function) so the fallback path is exercised; it must fault both a direct call and a path `run_all` reaches.

The `run_all` own-failure and injected-sink tests go through `run_value_int`, which fires `FEATURE_GROUP_MATCHED`, `FEATURE_GROUP_CALCULATE_FEATURE` and `VALIDATE_OUTPUT_FEATURE` but never `INPUT_DATA_LOAD`, `JOIN` or `VALIDATE_INPUT_FEATURE`. A host whose extender wraps only unfired hooks cannot pass them, whatever `own_failure()` does (a missing warning, a bare `DID NOT RAISE`, or an empty sink probe), so it overrides `test_contract_run_all_own_failure_falls_back_when_raise_on_error_false` and `test_contract_run_all_own_failure_propagates_when_raise_on_error_true` by name, plus `test_contract_run_all_emits_into_the_exact_injected_sink` when `has_backend_sink()` is `True` (and its real-worker counterpart when `supports_real_worker_sink()` is). In the overrides, run `run_csv_feature` (fires `INPUT_DATA_LOAD`) or `run_two_features` (fires `VALIDATE_INPUT_FEATURE`) instead, both in `mloda.testing.extenders.runners`; `run_joined_features` (same module) fires `JOIN` (an inner join of two PyArrow sources); both it and `run_two_features` forward `parallelization_modes` and `flight_server` like `run_csv_feature`, so under `MULTIPROCESSING` the `JOIN` and `VALIDATE_INPUT_FEATURE` hooks run in a real worker. A host's own runner needs module-level feature groups there, since core refuses to send a function-local one (or a link to one) to a worker; give them a distinctive class-name prefix (as the `MlodaTesting*` fixtures do), since the default matcher also matches the class name.

An autouse fixture enters `verified_context` with `context_identity` around every test (a no-op when empty); host tests that build their own hook context use `contract_context()`, which carries the same identity. The OTel and OpenLineage mixin tests still build some hook contexts without it, so an identity-gated extender cannot host those mixins yet.

The mixin pins:

- `wraps()` returns only known hooks, and the exact set when `expected_hooks()` is declared
- the `raise_on_error` default, and that it is configurable through the constructor
- a call returns the wrapped result unchanged, with or without an ambient `HookContext`
- a wrapped failure propagates and runs the wrapped function exactly once
- the extender's own failure falls back with a warning when `raise_on_error` is `False`, and propagates when `True`
- own failure is contained: a chained extender still runs, and a `run_all` round trip still completes with the warning-only fallback
- with `raise_on_error=True` (breaking-only hosts included), the extender's own failure propagates out of a `run_all` round trip
- the extender survives a pickle round trip, and a pickled copy still wraps a call
- when `supports_pickled_sink_capture()` is `True`: a picklable injected sink survives pickling and the pickled copy still emits into it
- when `supports_unpicklable_sink_degrade()` is `True`: an unpicklable injected sink is dropped and warned about exactly once across repeated pickling, and the resulting copy still wraps a call
- `run_all` round trips (one success, one wrapped failure)
- a `run_all` round trip through a CSV load (`run_csv_feature`, fires `INPUT_DATA_LOAD`) returns the loaded column unchanged, so an extender that changes the loaded CSV data fails it
- when `has_backend_sink()` is `True`: an unconfigured extender emits nothing against ambient sink configuration, both on a direct call and in `run_all`; `use_sdk_defaults=True` resolves the sink from that same ambient configuration and emits into it, not merely looks it up (skipped when `ambient_sink_captured` returns `None`); an injected sink ignores ambient configuration entirely; a `run_all` under `SYNC` or `THREADING` (no real subprocess, so core never pickles the extender) reaches the exact injected sink object, with no drop-warning logged; a pickled `use_sdk_defaults=True` copy still resolves the sink from ambient configuration. Extenders with no external sink inherit `has_backend_sink()` returning `False` and skip these tests
- when `supports_real_worker_sink()` is `True`: a real spawned `MULTIPROCESSING` worker reaches the exact injected sink (a marker file exists), with no drop-warning logged; if `supports_unpicklable_sink_degrade()` is also `True`, an unpicklable injected sink degrades gracefully there too (the run still completes, and a drop-warning is logged from the parent process, where core's own preflight pickle check runs before the worker is ever spawned)

### OtelExtenderTestMixin

Install `mloda-testing[otel]`. Host provides `extender_class` and `make_otel_extender(tracer_provider, *, raise_on_error=None)`, and optionally `expected_span_names`, `expected_span_name(context)` (per-context name, defaults to the `expected_span_names()` entry for the hook; override it when a name depends on the context, such as `calculate <FeatureGroup>`) and `trace_id_from_run_id` (the run_id-to-trace-id mapping; return `None` to skip the derivation test). It supplies `make_extender`, `own_failure`, and the sink-resolution hooks (`has_backend_sink`, `ambient_sink_environment`, `sink_resolution_spy`, `ambient_sink_captured`), so a host needs no extra code for the sink-resolution tests. It also sets `supports_pickled_sink_capture()` to `True` and supplies `injected_sink_capture`, exercising the picklable injected sink preservation.

`own_failure` and `pickled_copy_environment` are overridable on both backend mixins. The OTel default faults `TracerProvider.get_tracer`; an extender that caches its tracer at construction must override `own_failure`. The OpenLineage default faults `OpenLineageClient.emit`. A host whose pickled copy would resolve a real sink overrides `pickled_copy_environment`.

```python
from opentelemetry.sdk.trace import TracerProvider

from mloda.testing.extenders.otel import OtelExtenderTestMixin


class TestMyOtelExtenderContract(OtelExtenderTestMixin):
    @classmethod
    def extender_class(cls) -> type[MyOtelExtender]:
        return MyOtelExtender

    def make_otel_extender(
        self, tracer_provider: TracerProvider, *, raise_on_error: bool | None = None
    ) -> MyOtelExtender:
        if raise_on_error is None:
            return MyOtelExtender(tracer_provider=tracer_provider)
        return MyOtelExtender(tracer_provider=tracer_provider, raise_on_error=raise_on_error)
```

The mixin pins:

- one span per call
- per-hook span names, when `expected_span_names()` is declared
- a wrapped failure marks the span `ERROR` and logs a WARNING with the extender name and exception type, without leaking the exception message
- the carrier parents the span; without a carrier and without a run root, the trace id derives from `run_id`
- `run_all` spans form one well-formed trace: one trace id, exactly one span without a parent (the root) and every other span's parent among the spans; the calculate span is found by `mloda.operation.name == "calculate"`
- an interrupt (`BaseException`) still marks the span `ERROR` without leaking the exception message
- when the host wraps `INPUT_DATA_LOAD` (else skipped): the query string and user information of a nested load's raw data access (`args[0]`) never reach any span attribute; a load nested inside a calculate call is a child of that calculate span

A host that must record the raw URI overrides `test_otel_input_data_load_query_string_never_reaches_span_attributes` by name.

Helpers: `make_span_capture`, `make_picklable_span_capture`, `single_span`, `single_span_attributes`, `inject_parent_carrier`, `RebuildingSpanCaptureProvider`, `FileSpanExporter` (writes finished span names to a marker file, for `make_real_worker_extender_and_marker`; `records=True` on either writes one JSON record per span, with ids and attributes, instead of a bare name), `read_span_records` (parses those records back).

### OpenLineageExtenderTestMixin

Install `mloda-testing[openlineage]`. Host provides `extender_class` and `make_openlineage_extender(client, *, raise_on_error=None)`. It supplies `make_extender`, `own_failure`, and the sink-resolution hooks (`has_backend_sink`, `ambient_sink_environment`, `sink_resolution_spy`, `ambient_sink_captured`), so a host needs no extra code for the sink-resolution tests. It also sets `supports_pickled_sink_capture()` to `True` and supplies `injected_sink_capture`, exercising the picklable injected sink preservation. Optional: `calculate_run_events(events)` returns only the events of the calculate runs (the default drops the root-job parent run events); override it in a host that also emits other runs, such as nested validation runs, so the `run_all` tests that count COMPLETE events or inputs ignore them. A subclass of `OpenLineageExtender` can call `assert_openlineage_extender_seams(MyExtender, OpenLineageExtender)` to check that the subclass body keeps the declared seams and touches no other private base member (`MyExtender` must be constructible with no arguments).

```python
from openlineage.client.client import OpenLineageClient

from mloda.testing.extenders.openlineage import OpenLineageExtenderTestMixin


class TestMyOpenLineageExtenderContract(OpenLineageExtenderTestMixin):
    @classmethod
    def extender_class(cls) -> type[MyOpenLineageExtender]:
        return MyOpenLineageExtender

    def make_openlineage_extender(
        self, client: OpenLineageClient, *, raise_on_error: bool | None = None
    ) -> MyOpenLineageExtender:
        if raise_on_error is None:
            return MyOpenLineageExtender(client=client)
        return MyOpenLineageExtender(client=client, raise_on_error=raise_on_error)
```

The mixin pins:

- a run emits START then COMPLETE, or START then FAIL
- the START event precedes the wrapped call
- the COMPLETE event carries one output per feature name
- an `Exception` from the wrapped call ends in a FAIL event, any other `BaseException` (an interrupt) in an ABORT event
- an emit failure never masks the wrapped exception or corrupts the result
- no event ever leaks the exception message, and the WARNING for a wrapped failure names the exception type, never the message
- the parent facet ties the run to the ambient `run_id`
- a nested `INPUT_DATA_LOAD` call becomes an input, on both COMPLETE and FAIL, when the extender wraps that hook; the input is attributed before the load runs, so a failing load still appears on the FAIL event, and inputs mean attempted reads
- a nested `INPUT_DATA_LOAD` becomes an input named from the context's `data_access_identity` (an `s3://bucket/key` identity is namespace `s3://bucket`, name `key`), and the URI query string and user information of its raw data access (`args[0]`) reach no event, when the extender wraps that hook
- the calculate context's declared `input_features` become inputs too, on both COMPLETE and FAIL, so a host must report them
- a START emit failure under warning-only mode never prevents the wrapped call from running
- `run_all` emits exactly one parent START and one COMPLETE for the root job, and every step's parent runId equals that run id; the parent run carries the `mlodaPlan` facet with the plan id and structure hash; a failing run gives FAIL; preparing once and running twice gives two parent runs with the same plan id and structure hash
- a calculate step's runId equals `step_run_id(...)` (see [Run correlation](#run-correlation))
- with a recording OpenTelemetry span current, the step's START and terminal events carry an `mlodaTrace` run facet with that span's ids, and none otherwise (see [Traces and lineage](12-export-telemetry.md#traces-and-lineage))

Record the context's `data_access_identity`, not `args[0]`: a host that records the raw data access fails, and so does one that drops the identity, since inputs mean attempted reads. The rule is enforced on every host wrapping `INPUT_DATA_LOAD` because a presigned URL, a SAS token, or `user:password@` in a published dataset name is a credential leak; a host that must publish the raw URI overrides `test_openlineage_input_data_load_query_string_never_reaches_events` by name.

`RecordingTransport`, `LockHoldingTransport`, `FileTransport` (writes emitted event types to a marker file, for `make_real_worker_extender_and_marker`) and `make_recording_client` live in `mloda.testing.extenders.openlineage`.

`make_hook_context` builds a `HookContext` for direct `__call__` tests.

### Asserting on warnings

Count only the WARNING records of the logger that emits the warning under test, never every record in `caplog.records`. Core logs some warnings once per process (for example "not registered in the plugin registry" under `MLODA_PLUGIN_REGISTRY_STRICT=warn`), so under pytest-xdist such a record appears only if no earlier test in the worker triggered it. For your extender's own warnings, filter by its module's logger:

```python
import logging

import pytest

import my_package.my_extender as my_extender_module


def _module_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if r.name == my_extender_module.__name__ and r.levelno == logging.WARNING
    ]
```

The warning-only fallback warning is logged by core's extender logger, not your module's, and names the extender: check that the name appears in a message, as `ExtenderContractTestMixin` does.

## Real Implementations

| File | Description |
|------|-------------|
| [otel_extender.py](https://github.com/mloda-ai/mloda-registry/blob/main/mloda/community/extenders/otel/otel_extender.py) | OpenTelemetry spans for calculate, validate, load and join hooks, metadata-only by default; a `mloda.run` root span per run (`trace_scope="plan"` adds a `mloda.plan` parent) and `calculate <FeatureGroup>` step spans; a load span nests under its enclosing calculate span when one is active, else under the run root, the carrier or `run_id`; inert until a `tracer_provider` is injected or `use_sdk_defaults=True` with an SDK tracer provider configured (`mloda-community-otel`) |
| [openlineage_extender.py](https://github.com/mloda-ai/mloda-registry/blob/main/mloda/community/extenders/openlineage/openlineage_extender.py) | OpenLineage RunEvents with schema, data-source and parent-run facets; inert until a `client` is injected or `use_sdk_defaults=True` (`mloda-community-openlineage`) |
| [audit_extender.py](https://github.com/mloda-ai/mloda-registry/blob/main/mloda/enterprise/extenders/audit/audit_extender.py) | Tenant-scoped audit record per calculation with an identity presence gate (`fail_closed=True` refuses an unidentified run at `on_run_start` or match time, before any feature is calculated, and allows a complete run identity that differs from the plan's) that also lists the identity and format of each distinct data load the call attempted (core's `data_access_identity`, recorded as given) and core's fallback flag per load; a sink failure after a successful calculation fails the run by default; every record is `record_version` 2 and carries a `policy_version`; `TeeAuditSink` writes a record to several sinks and `OtelLogAuditSink` (extra `mloda-enterprise[otel]`) emits it as an OpenTelemetry log record; constructed with `audit_path`, `manifest_path` and `signer`, `AuditExtender` seals each run itself from `on_run_complete`; each `run()` of a prepared session gets a fresh `run_id` and is audited and sealed as its own run; a `run_id` reused by hand writes records outside its seal, which `on_run_complete` counts as a seal failure and which `verify_ndjson_log_coverage` keeps reporting; do not run one prepared auto-sealing session concurrently, since a run sealed while another `run()` is still calculating leaves that run's later records outside the seal; a plan-time `fail_closed` refusal has no `run_id`, so its record is attributed to its `plan_id` (records carry a `plan_id` key); `on_plan_complete` seals it under its `plan_id` when the extender wrote a plan-time record for that plan, and a manual `seal_ndjson_runs` sweep seals it for an extender without a sealing config; auto-sealing can emit each new head to a `head_anchor` (`NdjsonHeadAnchor`), applies `seal_failure_policy` to a failed seal, and takes an opt-in `seal_index_path` seal index so a seal avoids re-reading the whole logs, and can rotate its segment after a seal past `segment_max_bytes` or `segment_max_age`; which otherwise seals a finished run manually into a signed, hash-chained manifest that `verify_ndjson_log` checks, `rotate_manifest_key` records a key change, `rotate_ndjson_segment` archives the logs into numbered segments and `verify_ndjson_segments` verifies them as one chain, `quarantine_damaged_lines` is the repair path for a torn log and `quarantine_from_rotation_entry` for a wrongly appended rotation entry; `Ed25519Signer` (extra `mloda-enterprise[ed25519]`) makes seals non-repudiable and verifiable with the public key alone, and the unchanged `ManifestSigner` protocol lets a KMS-backed signer plug in later (none ships yet) (`mloda-enterprise-audit`, license required) |
| [lineage_extender.py](https://github.com/mloda-ai/mloda-registry/blob/main/mloda/enterprise/extenders/lineage/lineage_extender.py) | Used instead of `OpenLineageExtender`: adds column lineage, declared masking, validator-outcome assertions and a `mloda` run facet with a structure hash (`mloda-enterprise-lineage`, extra `mloda-enterprise[openlineage]`, license required) |
| [contract.py](https://github.com/mloda-ai/mloda-registry/blob/main/mloda/testing/extenders/contract.py) | Extender contract test mixin (mloda-testing) |
| [test_composite_extender.py](https://github.com/mloda-ai/mloda/blob/main/tests/test_plugins/extender/test_composite_extender.py) | Chaining tests |
