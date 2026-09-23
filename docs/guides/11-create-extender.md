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

## Chaining and Error Handling

Multiple extenders for the same hook chain automatically (sorted by priority, lower first).

Extender failures are breaking by default, both for a single extender and in a chain: the exception propagates and the calculation fails. An extender opts into warning-only behavior by setting `raise_on_error = False` (commonly a constructor argument). Its failure is then logged as a warning and the wrapped function still runs. Non-critical or observability extenders should pass `False`.

Only the extender's own failure is caught. An exception raised by the wrapped function always propagates, and the wrapped function is never run twice.

An extender that refuses a call (a gate) must keep `raise_on_error = True`: a warning-only extender that raises before delegating is logged, and the wrapped function runs anyway.

## Pickle Compatibility

Only needed with `ParallelizationMode.MULTIPROCESSING`. Avoid unpicklable instance variables (locks, tracers, connections). Use class-level storage or create resources lazily in `__call__()`.

Both extenders trial-pickle an injected sink (`client` for `OpenLineageExtender`, `tracer_provider` for `OtelExtender`) whenever a copy is made (worker processes under `MULTIPROCESSING`). A picklable sink is pickled as-is and survives into the worker. An unpicklable one is dropped instead: logged once per instance at WARNING, naming the sink and the underlying pickle exception's type, and the resulting copy is inert unless `use_sdk_defaults=True`. With `use_sdk_defaults=True`, `OtelExtender` resolves a provider installed in the worker via `child_bootstrap` passed to `run_all`, and `OpenLineageExtender` builds its own client lazily per worker.

mloda terminates `MULTIPROCESSING` workers with no teardown, so a buffered sink loses events there: OpenLineage `async_http` or kafka transports, and an OTel `BatchSpanProcessor` (`BatchLogRecordProcessor` for `OtelLogAuditSink`) installed by `child_bootstrap`. Use a synchronous sink instead under `MULTIPROCESSING`: OpenLineage `http`, `console`, or `file` transport (injected or via `OPENLINEAGE_CONFIG`), or OTel `SimpleSpanProcessor` (`SimpleLogRecordProcessor` for `OtelLogAuditSink`). The `user_hash_key` of `OtelLogAuditSink` travels in the pickled sink under `MULTIPROCESSING`, so treat that pickle as secret material and load the key from a secret store or env var rather than hard-coding it.

## Emitting on the calculation thread

Extender code runs inline with the wrapped call: a blocking sink stalls every call, and with `raise_on_error=True` a sink failure fails the run. For OpenLineage under SYNC and THREADING, prefer the `async_http` transport or a short timeout (via `OPENLINEAGE_CONFIG` or `OPENLINEAGE__TRANSPORT__*`), and keep `raise_on_error=False` for observability. Under `MULTIPROCESSING`, see [Pickle Compatibility](#pickle-compatibility) for the synchronous-sink requirement.

## Sink Resolution

Adding an extender opts a pipeline into instrumentation, not into ambient configuration. `use_sdk_defaults` is the explicit opt-in for that. Resolution order, strict:

1. An injected client/provider wins. No other sink is resolved alongside it.
2. Else `use_sdk_defaults=True` delegates fully to the vendor SDK's own resolution (globals, env vars, config files, and a console fallback where the SDK has one).
3. Else the extender is inert: no vendor configuration consulted, no backend constructed, nothing emitted. The wrapped call still runs and its result is returned unchanged. Logged once per instance at WARNING, including after a pickle round trip that drops an injected sink.

`mloda-community-otel` ships `opentelemetry-api` only, whose default tracer provider is a no-op with no console fallback, so `OtelExtender(use_sdk_defaults=True)` exports nothing until an SDK `TracerProvider` with an exporter is configured. Install `opentelemetry-sdk` and call `opentelemetry.trace.set_tracer_provider` (under `MULTIPROCESSING`, in each worker via `child_bootstrap`). With only the API default present it logs one WARNING per instance (per copy after pickling); to run without an SDK on purpose, inject `opentelemetry.trace.NoOpTracerProvider()` as `tracer_provider`. Any extender whose `use_sdk_defaults` can resolve to a vendor no-op should warn once the same way.

## Usage

```python
from mloda.user import mloda

results = mloda.run_all(features=["my_feature"], function_extender={MyExtender(), OtherExtender()})
```

## Hook context for data loads

`INPUT_DATA_LOAD` runs nested inside the active `FEATURE_GROUP_CALCULATE_FEATURE` call, but an extender may wrap it alone: the calculation context is activated whenever either hook has an extender registered, so a data-load wrapper still fires when nothing wraps `FEATURE_GROUP_CALCULATE_FEATURE`.

Its `HookContext` inherits the enclosing calculation's feature-group and feature identity fields (`feature_group_class`, `feature_group_version`, `plugin_version`, `feature_names`, `input_features`), then adds `data_access_identity` and `data_access_format` for the input being read. The enclosing calculation context does not carry those data-access fields, and `rows_in` is left unset here.

The `data_access_identity` that core supplies on the context is not guaranteed to be credential-free. Core strips URI user information such as `user:password@` but retains a URI query string. If query parameters can contain SAS tokens, presigned signatures, or other secrets, do not persist this field without additional redaction.

`AuditExtender` strips the query string, fragment, `;` and `&` parameters and user information from URI-shaped identities, best effort. Any other value, such as a keyword DSN or an object's repr, is recorded as core supplies it, so the field is not credential-free, and a sealed log cannot be redacted afterwards. OpenLineage and `OtelExtender` sanitize the identity the same way (via the shared `resolve_data_access_identity`), so dataset names and `mloda.data_access.identity` span attributes taken from data-access identities carry no query, and datasets emitted earlier with one do not join to the new names. The sanitizer only cleans `scheme://`-shaped identities; anything else is left as given. Known limits, kept by decision: an HTTP source whose query is its identity collapses to one dataset, and identities that differ only in user information collapse too, most visibly Azure `abfss://container@account...` (also `abfs`, `wasb`, `wasbs`), so containers on one account with the same path become one dataset (the sanitized string is also the dedupe key and dataset identity). Keeping the container would need a scheme allowlist or a userinfo heuristic in the shared credential stripping (a heuristic would leak bare-token userinfo such as `https://TOKEN@host/`) and would change recorded audit identities, so it is not done. `data_access_identity` and `data_access_format` are index-aligned lists with one entry per distinct (identity, format) pair the call attempted, in first-seen order: repeats collapse, a load without an identity is omitted, and `[]` means none. An absent key means not recorded, since keys may be added within `record_version` 1.

Feature names are published as given wherever they become dataset names: declared input features and outputs in `OpenLineageExtender`, and also column-lineage edges and validation datasets in `LineageFacetsExtender`. A consumer's input name then equals its producer's output name, which is what joins the lineage graph. Only data-access identities are sanitized; a feature name is an author- or caller-chosen identifier, so never put a secret in one. A feature named like a load identity is merged with that load's dataset only when its name equals the sanitized identity, so not while the name still carries the stripped query.

### Declared attributes

`OtelExtender` reads a `declared_attributes(features)` classmethod off the class owning the wrapped call (the `FeatureGroup` for a calculate hook, the reader class for a load hook), passing the call's `FeatureSet` (may be `None`). Its returned mapping is set as `mloda.declared.<key>` on the span, before the wrapped call runs, on calculate and load spans only, never on validate. Metadata only: do not echo option values back, since `FeatureSet` options can carry credentials. A missing method is a silent no-op. A raise or a non-mapping return is contained: logged at WARNING on each failing call (extender and exception type, never the message), then skipped; the wrapped call still runs.

## Verified run context

Use `mloda.steward.verified_context()` around the run call when an extender needs server-verified tenant, project, or principal identity. These values populate `tenant_id`, `project_id`, and `principal` on every `HookContext` created for that run and cannot be overridden through feature `Options`.

```python
from mloda.steward import verified_context
from mloda.user import mloda

with verified_context(tenant_id="tenant-42", project_id="project-7", principal="service-account"):
    results = mloda.run_all(features=["my_feature"], function_extender={MyExtender()})
```

`AuditExtender(sink, fail_closed=True)` (`mloda-enterprise-audit`) turns a missing required identity into a refusal: it writes the deny record, then raises `IdentityRequiredError` before the wrapped call runs, so no feature is calculated. It sets `priority = 0` so the gate runs outermost; an extender with a lower priority would still run outside it. It gates two hooks:

- `FEATURE_GROUP_MATCHED` reads `verified_context` at plan time, so `prepare`, `explain` and `diagnose` need the scope too, not only `run`. It writes a record (which `OtelLogAuditSink` also emits) only when it refuses, before any feature is calculated. Without `fail_closed`, a missing identity denies at calculate and the run proceeds. No allow event is written at match; the calculate record is the allow event.
- `FEATURE_GROUP_CALCULATE_FEATURE` reads it at `run()` or `stream_run()` time. It is the only gate for an extender passed only to `run()`, and it catches a session prepared inside the scope but run outside it.

The refusal record has `decision="deny"`, `status="error"` and an `error_type` naming `IdentityRequiredError`; a match-time refusal leaves the feature-group fields empty. It is only as durable as the sink, so use a synchronous one such as `NdjsonAuditSink` under `MULTIPROCESSING` (see [Pickle Compatibility](#pickle-compatibility)). The constructor rejects `fail_closed=True` with `raise_on_error=False` or an empty `required_identity`; do not change `raise_on_error` or `fail_closed` afterwards, as only the constructor checks them. Two paths still fail open: a call made outside a core run (no hook context), and an extender that registry strict mode `strict` (`MLODA_PLUGIN_REGISTRY_STRICT`) drops as unregistered.

Every record carries `policy_version`: the argument of that name (a non-blank string), else a 12 hex character fingerprint of the constructor-supplied gate (sorted `required_identity` plus `fail_closed`), which does not track code changes. The key is additive within `record_version` 1.

`TeeAuditSink(*sinks)` writes each record to every sink in order; the first failure propagates and later sinks are skipped, so put the durable `NdjsonAuditSink` first. It raises `ValueError` for no sinks or a sink without a callable `write`.

`OtelLogAuditSink()` (or `OtelLogAuditSink(user_hash_key=b"...")`, see `user.hash` below) emits one OpenTelemetry log record per audit record, so allow and deny decisions reach a Collector. It needs the extra `mloda-enterprise[otel]` (`opentelemetry-api>=1.37,<2`); the package loads without it and only constructing the sink raises `ImportError` naming the extra. Allow is INFO, deny is WARN, the body is the decision and the timestamp is `event_time`. Attributes: `mloda.audit.decision`, `mloda.audit.deny_reason`, `mloda.audit.policy_version`, `mloda.audit.hook`, `mloda.run.id`, `mloda.tenant.id`, `mloda.project.id`, `user.hash`, `mloda.feature_group.name`, `mloda.feature.names`, `error.type`; blank or absent values are omitted.

`user.hash` is the sha256 hex of the principal by default: pseudonymous, not anonymous, so hash the principal to join it to the NDJSON record: `hashlib.sha256(record["principal"].encode("utf-8")).hexdigest()`. An unkeyed hash of a low-entropy principal (email, username) can be confirmed offline by dictionary and is identical across deployments, so `OtelLogAuditSink(user_hash_key=b"...")` (bytes, at least 32) exports HMAC-SHA256 hex instead; still pseudonymous, and one key per deployment still links a principal across its tenants (use a key per tenant to avoid that). Use a key of its own, never the manifest signing key. Join with `hmac.new(key, record["principal"].encode("utf-8"), hashlib.sha256).hexdigest()` against the NDJSON `principal`, which stays the identifying copy. Switching variants or rotating the key changes every `user.hash`, so older records no longer join. `mloda.run.id` joins to `OtelExtender` spans. Data-access identities, feature values and exception messages are never included. Emitting is best effort: a failure is logged at WARNING and never fails the run. The sink looks up the global logger provider on every write and logs one WARNING per process while only the API default provider is installed. Under `MULTIPROCESSING` a spawned worker has no SDK logger provider unless `child_bootstrap` installs one (see [Pickle Compatibility](#pickle-compatibility)), and a synchronous exporter blocks the calculation thread. The OTel copy is not sealed or tamper-evident, so keep the NDJSON log and its manifest as the retained record and compose them:

```python
from mloda.enterprise.extenders.audit import AuditExtender, NdjsonAuditSink, OtelLogAuditSink, TeeAuditSink

sink = TeeAuditSink(NdjsonAuditSink("audit.ndjson"), OtelLogAuditSink())
extender = AuditExtender(sink, fail_closed=True)
```

## Lineage facets

`LineageFacetsExtender` (`mloda-enterprise-lineage`) is used instead of `OpenLineageExtender`, not next to it. It needs the extra `mloda-enterprise[openlineage]`; without it the extender is not registered. Beyond the community events it adds:

- `columnLineage` on each output dataset: DIRECT edges from the step's declared input features, or, for a root step that declares its source column, from the one dataset it loaded.
- `masking` on those edges, only when declared (see below).
- `dataQualityAssertions` on validation runs: one assertion named after the validator, `success` true or false.
- a `mloda` run facet: `featureGroupVersion`, `pluginVersion`, `computeFramework`, `maskedFeatures` and `structureHash`.

Masking is declared, never inferred: set the class attribute `masking = True` on the feature group, or the option `masking=True` in the feature's own `context`. It must be the boolean `True`; a `group` key, the string `"true"` and a context key forwarded from another step do not count. It is unrelated to core's `mask`.

Column-lineage edges are step-level declared inputs. A step with several outputs cannot say which input feeds which, so every output lists every input (an over-approximation) and the transformation `description` reads `step-level declared inputs`.

A root step has an edge only when it declares the source column (a feature name equalling a column is common but not guaranteed). Declare it with the option `lineage_source_column` in the feature's own `context` (a column name, or `True` for the feature name) or the class attribute `lineage_source_column` (a dict from feature name to column, or `True` for every feature name); a valid option wins. The edge runs from the loaded dataset, named as its input dataset is: the sanitized `data_access_identity` (see [Hook context for data loads](#hook-context-for-data-loads)); distinct sources with equal identities, such as dict credentials with the same keys, count as one dataset. The column is declared, never verified against the data. A `group` key, a forwarded key, `False`, an empty string and any other type do not count, and the class attribute then still applies. A root step that loads no dataset (`DataCreator`) or several distinct ones has no edge, and a step with declared inputs keeps only its input edges.

Validation is reported only for a feature group that overrides `validate_input_features` or `validate_output_features`. Each overridden validator is its own run per step, a sibling of the calculate run and both under the root run (job `<feature group>.<method>`, START then COMPLETE, FAIL or ABORT), so it adds one extra run and two events per step. The terminal event carries one input dataset per validated feature. Validate-input is skipped when the step has no data yet (root steps); when data exists it runs and the validated datasets fall back to the feature names if no inputs are declared. Core's own `EmptyResultError` and `DataTypeValidator` failures happen outside the hook, so they produce no assertion.

A failing validator is re-raised and marks every validated dataset `success=false`, because the failing one is unknown. Its message can reach the WARNING log, as a calculate failure message already does, but never an event.

Validation runs carry the parent facet only, not the `mloda` facet. Tie one to a calculate run through the job name and the run id.

Validate-output assertions ride on input datasets of the `<feature group>.validate_output_features` job, the only spec-legal home for `dataQualityAssertions`.

`structureHash` is the sha256 of the feature group class, versions, compute framework, feature names, declared inputs, masked features and, for a root step, the declared source column of each feature (option or class attribute). It covers the declaration, not whether an edge was emitted. It holds no run id or time, so it is stable across runs. It also changes with the feature group source and the mloda version, because both are part of `feature_group_version`.

An option (`masking`, `lineage_source_column`) counts only when the step declares it in its own `context` and its consumer does not hold the same key with an equal value. A key core marks inherited (`propagate_context_keys`, `inherit_context_keys`) never counts. Core also cannot tell a step's own declaration from a consumer's value that reached it through an Options object shared with the step or copied by core, or from an equal declaration of its own, so none of these is attributed to the step: it reports no masking and no root edge from the option. `maskedFeatures` is therefore a declared lower bound. When several consumers request the same feature, core keeps one of them, so the step is judged against that one and `structureHash` can then differ between runs. Declare `masking = True` (or `lineage_source_column`) on the feature group class for a guarantee that does not depend on any consumer.

The `mloda` facet's schema URL points at its module in this repository; it is not a hosted JSON schema.

## Testing

`mloda-testing` ships three test mixins plus helpers (`make_hook_context`, `run_value_int`, `failing_feature_group`) so every extender's test suite exercises the same shared behavior:

- `ExtenderContractTestMixin` (`mloda.testing.extenders.contract`), for every extender
- `OtelExtenderTestMixin` (`mloda.testing.extenders.otel`, install `mloda-testing[otel]`), for extenders that emit OTel spans
- `OpenLineageExtenderTestMixin` (`mloda.testing.extenders.openlineage`, install `mloda-testing[openlineage]`), for extenders that emit OpenLineage RunEvents

The OTel and OpenLineage mixins both enforce the same observability mandate: a wrapped failure is logged at WARNING with the extender name and message, but the message itself never reaches a span or an event.

### ExtenderContractTestMixin

Required host hooks: `extender_class`, `make_extender`, `own_failure`. Optional: `raise_on_error_default`, `expected_hooks`, `pickled_copy_environment`, `supports_warning_only` (return `False` for a host with no `raise_on_error=False` mode; the `run_all` wrapped-failure test then uses the default `raise_on_error`), `context_identity` (a dict of `tenant_id` / `project_id` / `principal` for an identity-gated extender; carried by every contract context and `run_all`; defaults to empty), `has_backend_sink` (return `True` for an extender with an external sink, and override `ambient_sink_environment` plus `sink_resolution_spy`, and optionally `ambient_sink_captured(spy)`, which returns what the ambient sink(s) the spy handed out captured once the call is done; its base default `None` opts out of the emission check, for a host that cannot expose a probe; `make_unconfigured_extender`/`make_sdk_defaults_extender` default to `extender_class()()` and `extender_class()(use_sdk_defaults=True)`), `supports_pickled_sink_capture` plus `injected_sink_capture` (`True` when a picklable injected sink survives pickling; defaults to `False`), `supports_unpicklable_sink_degrade` plus `make_unpicklable_sink_extender` (`True` when the host trial-pickles its sink and drops+warns instead of hard-failing pickling; defaults to `False`), `sink_noun` (the word the drop-warning uses for the sink, e.g. `"tracer_provider"`; `None` skips noun-specific assertions), `unpicklable_sink_failure_type` (the pickle-failure exception type name the drop-warning names; defaults to `"TypeError"`, correct for a sink holding a `threading.Lock`).

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

The `run_all` own-failure and injected-sink tests go through `run_value_int`, which fires `FEATURE_GROUP_MATCHED`, `FEATURE_GROUP_CALCULATE_FEATURE` and `VALIDATE_OUTPUT_FEATURE` but never `INPUT_DATA_LOAD`, `JOIN` or `VALIDATE_INPUT_FEATURE`. A host whose extender wraps only unfired hooks cannot pass them, whatever `own_failure()` does (a missing warning, a bare `DID NOT RAISE`, or an empty sink probe), so it overrides `test_contract_run_all_own_failure_falls_back_when_raise_on_error_false` and `test_contract_run_all_own_failure_propagates_when_raise_on_error_true` by name, plus `test_contract_run_all_emits_into_the_exact_injected_sink` when `has_backend_sink()` is `True` (and its real-worker counterpart when `supports_real_worker_sink()` is). In the overrides, run `run_csv_feature` (fires `INPUT_DATA_LOAD`) or `run_two_features` (fires `VALIDATE_INPUT_FEATURE`) instead, both in `mloda.testing.extenders.runners`; no shipped runner fires `JOIN`, so a host wrapping it needs its own `run_all`.

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
- when `has_backend_sink()` is `True`: an unconfigured extender emits nothing against ambient sink configuration, both on a direct call and in `run_all`; `use_sdk_defaults=True` resolves the sink from that same ambient configuration and emits into it, not merely looks it up (skipped when `ambient_sink_captured` returns `None`); an injected sink ignores ambient configuration entirely; a `run_all` under `SYNC` or `THREADING` (no real subprocess, so core never pickles the extender) reaches the exact injected sink object, with no drop-warning logged; a pickled `use_sdk_defaults=True` copy still resolves the sink from ambient configuration. Extenders with no external sink inherit `has_backend_sink()` returning `False` and skip these tests
- when `supports_real_worker_sink()` is `True`: a real spawned `MULTIPROCESSING` worker reaches the exact injected sink (a marker file exists), with no drop-warning logged; if `supports_unpicklable_sink_degrade()` is also `True`, an unpicklable injected sink degrades gracefully there too (the run still completes, and a drop-warning is logged from the parent process, where core's own preflight pickle check runs before the worker is ever spawned)

### OtelExtenderTestMixin

Install `mloda-testing[otel]`. Host provides `extender_class` and `make_otel_extender(tracer_provider, *, raise_on_error=None)`, and optionally `expected_span_names` and `trace_id_from_run_id` (the run_id-to-trace-id mapping; return `None` to skip the derivation test). It supplies `make_extender`, `own_failure`, and the sink-resolution hooks (`has_backend_sink`, `ambient_sink_environment`, `sink_resolution_spy`, `ambient_sink_captured`), so a host needs no extra code for the sink-resolution tests. It also sets `supports_pickled_sink_capture()` to `True` and supplies `injected_sink_capture`, exercising `OtelExtender`'s picklable-injected-`tracer_provider` preservation.

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
- a wrapped failure marks the span `ERROR` without leaking the exception message
- the carrier parents the span; without a carrier, the trace id derives from `run_id`
- `run_all` spans share one trace id, and the check requires at least two spans, one of them the declared calculate span name
- an interrupt (`BaseException`) still marks the span `ERROR` without leaking the exception message
- when the host wraps `INPUT_DATA_LOAD` (else skipped): the query string of a nested load identity never reaches any span attribute; a load nested inside a calculate call is a child of that calculate span

Helpers: `make_span_capture`, `make_picklable_span_capture`, `single_span`, `single_span_attributes`, `inject_parent_carrier`, `RebuildingSpanCaptureProvider`, `FileSpanExporter` (writes finished span names to a marker file, for `make_real_worker_extender_and_marker`).

### OpenLineageExtenderTestMixin

Install `mloda-testing[openlineage]`. Host provides `extender_class` and `make_openlineage_extender(client, *, raise_on_error=None)`. It supplies `make_extender`, `own_failure`, and the sink-resolution hooks (`has_backend_sink`, `ambient_sink_environment`, `sink_resolution_spy`, `ambient_sink_captured`), so a host needs no extra code for the sink-resolution tests. It also sets `supports_pickled_sink_capture()` to `True` and supplies `injected_sink_capture`, exercising `OpenLineageExtender`'s picklable-injected-client preservation. Optional: `calculate_run_events(events)` (identity by default) returns only the events of the calculate runs; override it in a host that also emits other runs, such as nested validation runs, so the `run_all` tests that count COMPLETE events or inputs ignore them.

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
- no event ever leaks the exception message
- the parent facet ties the run to the ambient `run_id`
- a nested `INPUT_DATA_LOAD` call becomes an input, on both COMPLETE and FAIL, when the extender wraps that hook; the input is attributed before the load runs, so a failing load still appears on the FAIL event, and inputs mean attempted reads
- the URI query string of a nested `INPUT_DATA_LOAD` identity reaches no event when the extender wraps that hook; user information is not pinned
- the calculate context's declared `input_features` become inputs too, on both COMPLETE and FAIL, so a host must report them
- a START emit failure under warning-only mode never prevents the wrapped call from running
- `run_all` events share one parent run id

Strip the query before recording: a host that drops the identity instead also fails, since inputs mean attempted reads. The rule is enforced on every host wrapping `INPUT_DATA_LOAD` because a presigned URL or SAS token in a published dataset name is a credential leak; a host that must publish the raw URI overrides `test_openlineage_input_data_load_query_string_never_reaches_events` by name.

`RecordingTransport`, `LockHoldingTransport`, `FileTransport` (writes emitted event types to a marker file, for `make_real_worker_extender_and_marker`) and `make_recording_client` live in `mloda.testing.extenders.openlineage`.

`make_hook_context` builds a `HookContext` for direct `__call__` tests.

## Real Implementations

| File | Description |
|------|-------------|
| [otel_extender.py](https://github.com/mloda-ai/mloda-registry/blob/main/mloda/community/extenders/otel/otel_extender.py) | OpenTelemetry spans for calculate, validate and load hooks, metadata-only by default; a load span nests under its enclosing calculate span when one is active, else falls back to the carrier or `run_id`; inert until a `tracer_provider` is injected or `use_sdk_defaults=True` with an SDK tracer provider configured (`mloda-community-otel`) |
| [openlineage_extender.py](https://github.com/mloda-ai/mloda-registry/blob/main/mloda/community/extenders/openlineage/openlineage_extender.py) | OpenLineage RunEvents with schema, data-source and parent-run facets; inert until a `client` is injected or `use_sdk_defaults=True` (`mloda-community-openlineage`) |
| [audit_extender.py](https://github.com/mloda-ai/mloda-registry/blob/main/mloda/enterprise/extenders/audit/audit_extender.py) | Tenant-scoped audit record per calculation with an identity presence gate (`fail_closed=True` refuses an unidentified run before any feature is calculated) that also lists the identity and format of each distinct data load the call attempted (URI query, fragment, parameters and user information stripped, best effort; other identities recorded as given, so not credential-free); a sink failure after a successful calculation fails the run by default; every record carries a `policy_version`; `TeeAuditSink` writes a record to several sinks and `OtelLogAuditSink` (extra `mloda-enterprise[otel]`) emits it as an OpenTelemetry log record; `seal_ndjson_runs` seals a finished run into a signed, hash-chained manifest that `verify_ndjson_log` checks, `rotate_manifest_key` records a key change, and `quarantine_damaged_lines` is the repair path for a torn log; `Ed25519Signer` (extra `mloda-enterprise[ed25519]`) makes seals non-repudiable and verifiable with the public key alone, and the unchanged `ManifestSigner` protocol lets a KMS-backed signer plug in later (none ships yet) (`mloda-enterprise-audit`, license required) |
| [lineage_extender.py](https://github.com/mloda-ai/mloda-registry/blob/main/mloda/enterprise/extenders/lineage/lineage_extender.py) | Used instead of `OpenLineageExtender`: adds column lineage, declared masking, validator-outcome assertions and a `mloda` run facet with a structure hash (`mloda-enterprise-lineage`, extra `mloda-enterprise[openlineage]`, license required) |
| [contract.py](https://github.com/mloda-ai/mloda-registry/blob/main/mloda/testing/extenders/contract.py) | Extender contract test mixin (mloda-testing) |
| [test_composite_extender.py](https://github.com/mloda-ai/mloda/blob/main/tests/test_plugins/extender/test_composite_extender.py) | Chaining tests |
