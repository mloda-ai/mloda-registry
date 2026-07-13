# Create an Extender Plugin

Add cross-cutting concerns (logging, tracing, metrics) to mloda pipelines.

## Decision Tree

```text
Q1: What do you want to wrap?
    Feature calculation → FEATURE_GROUP_CALCULATE_FEATURE
    Input validation   → VALIDATE_INPUT_FEATURE
    Output validation  → VALIDATE_OUTPUT_FEATURE

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

results = mloda.run_all(
    features=["my_feature"],
    function_extender={MyExtender(), OtherExtender()}
)
```

## Testing

```python
def test_my_extender():
    extender = MyExtender()
    assert ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE in extender.wraps()

    result = extender(lambda x, y: x + y, 1, 2)
    assert result == 3
```

## Real Implementations

| File | Description |
|------|-------------|
| [otel_extender.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/function_extender/base_implementations/otel/otel_extender.py) | OpenTelemetry |
| [test_composite_extender.py](https://github.com/mloda-ai/mloda/blob/main/tests/test_plugins/extender/test_composite_extender.py) | Chaining tests |
