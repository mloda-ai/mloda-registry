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

## Available Hooks

| Hook | When It Runs |
|------|--------------|
| `FEATURE_GROUP_CALCULATE_FEATURE` | Wraps `calculate_feature()` |
| `VALIDATE_INPUT_FEATURE` | Before calculation |
| `VALIDATE_OUTPUT_FEATURE` | After calculation |
| `FEATURE_GROUP_MATCHED` | Wraps feature group resolution |
| `INPUT_DATA_LOAD` | Wraps input data loading |
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

## Pickle Compatibility

Only needed with `ParallelizationMode.MULTIPROCESSING`. Avoid unpicklable instance variables (locks, tracers, connections). Use class-level storage or create resources lazily in `__call__()`.

## Usage

```python
from mloda.user import mloda

results = mloda.run_all(features=["my_feature"], function_extender={MyExtender(), OtherExtender()})
```

## Testing

`mloda-testing` ships three test mixins plus helpers (`make_hook_context`, `run_value_int`, `failing_feature_group`) so every extender's test suite exercises the same shared behavior:

- `ExtenderContractTestMixin` (`mloda.testing.extenders.contract`), for every extender
- `OtelExtenderTestMixin` (`mloda.testing.extenders.otel`, install `mloda-testing[otel]`), for extenders that emit OTel spans
- `OpenLineageExtenderTestMixin` (`mloda.testing.extenders.openlineage`, install `mloda-testing[openlineage]`), for extenders that emit OpenLineage RunEvents

### ExtenderContractTestMixin

Required host hooks: `extender_class`, `make_extender`, `own_failure`. Optional: `raise_on_error_default`, `expected_hooks`, `pickled_copy_environment`.

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

`extender_class` names the class under test. `make_extender` returns an instance wired to an in-memory backend, never a real network sink. `own_failure` makes the extender's own code fail (not the wrapped function) so the fallback path is exercised.

The mixin pins:

- `wraps()` returns only known hooks, and the exact set when `expected_hooks()` is declared
- the `raise_on_error` default, and that it is configurable through the constructor
- a call returns the wrapped result unchanged, with or without an ambient `HookContext`
- a wrapped failure propagates and runs the wrapped function exactly once
- the extender's own failure falls back with a warning when `raise_on_error` is `False`, and propagates when `True`
- own failure is contained: a chained extender still runs, and `run_all` still completes
- the extender survives a pickle round trip, and a pickled copy still wraps a call
- two `run_all` round trips (one success, one wrapped failure)

### OtelExtenderTestMixin

Install `mloda-testing[otel]`. Host provides `extender_class` and `make_otel_extender(tracer_provider, *, raise_on_error=None)`, and optionally `expected_span_names`. It supplies `make_extender` and `own_failure`.

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
- `run_all` spans share one trace id

Helpers: `make_span_capture`, `single_span`, `single_span_attributes`, `inject_parent_carrier`.

### OpenLineageExtenderTestMixin

Install `mloda-testing[openlineage]`. Host provides `extender_class` and `make_openlineage_extender(client, *, raise_on_error=None)`. It supplies `make_extender`, `own_failure`, and a `pickled_copy_environment` with OpenLineage disabled.

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
- a `BaseException` from the wrapped call still ends in a FAIL event
- an emit failure never masks the wrapped exception or corrupts the result
- no event ever leaks the exception message
- the parent facet ties the run to the ambient `run_id`
- a nested `INPUT_DATA_LOAD` call becomes an input, on both COMPLETE and FAIL, when the extender wraps that hook
- `run_all` events share one parent run id

`RecordingTransport` and `make_recording_client` live in `mloda.testing.extenders.openlineage` (moved from `mloda.community.extenders.openlineage.testing`).

`make_hook_context` builds a `HookContext` for direct `__call__` tests.

## Real Implementations

| File | Description |
|------|-------------|
| [otel_extender.py](https://github.com/mloda-ai/mloda-registry/blob/main/mloda/community/extenders/otel/otel_extender.py) | OpenTelemetry spans, metadata-only by default (`mloda-community-otel`) |
| [openlineage_extender.py](https://github.com/mloda-ai/mloda-registry/blob/main/mloda/community/extenders/openlineage/openlineage_extender.py) | OpenLineage RunEvents with schema, data-source and parent-run facets (`mloda-community-openlineage`) |
| [contract.py](https://github.com/mloda-ai/mloda-registry/blob/main/mloda/testing/extenders/contract.py) | Extender contract test mixin (mloda-testing) |
| [test_composite_extender.py](https://github.com/mloda-ai/mloda/blob/main/tests/test_plugins/extender/test_composite_extender.py) | Chaining tests |
