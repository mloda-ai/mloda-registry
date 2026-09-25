# Concept 7: Filter Engine

The filter engine handles filtering operations on data.

**What**: Component that implements filter operations (range, equal, regex, etc.).
**When**: Every compute framework needs a filter engine.
**Why**: Filters data during pipeline execution based on user criteria.
**Where**: Returned by `ComputeFramework.filter_engine()`.
**How**: Subclass `BaseFilterEngine`, implement all filter methods.

## Required Methods

| Method | Description |
|--------|-------------|
| `do_range_filter()` | Filter by min/max range |
| `do_min_filter()` | Filter >= value |
| `_apply_max_inclusive_filter()`, `_apply_max_exclusive_filter()` | Filter <= / < a threshold; the base `do_max_filter()` reads both parameter forms and calls them |
| `do_equal_filter()` | Filter == value |
| `do_regex_filter()` | Filter by regex pattern (unanchored search) |
| `do_categorical_inclusion_filter()` | Filter by set membership |
| `final_filters()` | Return True if filters applied at end |

A null or NaN row never passes a range, min, max or equal filter; categorical inclusion keeps null and NaN rows only when `values` contains `None` or NaN. See [Filter Data](https://mloda-ai.github.io/mloda/in_depth/filter_data/) for the full contract.

## Base Class Methods (override optional)

| Method | Description |
|--------|-------------|
| `applicable_filters()` | Return filters whose columns exist in output. Provided by `BaseFilterEngine` with a default that matches filter columns against the FeatureSet output. Override only if your framework needs custom column-matching logic. |

## Complete Example

```python
import math
import re
from typing import Any, Callable

from mloda.provider import BaseFilterEngine


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


class MyFilterEngine(BaseFilterEngine):
    """Filter engine for MyFramework (row-wise list[dict])."""

    @classmethod
    def final_filters(cls) -> bool:
        return True

    @classmethod
    def _keep(cls, data: Any, column: str, predicate: Callable[[Any], bool]) -> Any:
        # A null or NaN row never passes a comparison.
        return [row for row in data if not _is_missing(row.get(column)) and predicate(row[column])]

    @classmethod
    def do_range_filter(cls, data, filter_feature) -> Any:
        min_val, max_val, is_exclusive = cls.get_min_max_operator(filter_feature)
        if min_val is None or max_val is None:
            raise ValueError(f"Filter parameter {filter_feature.parameter} not supported")
        if is_exclusive:
            return cls._keep(data, filter_feature.name, lambda v: min_val <= v < max_val)
        return cls._keep(data, filter_feature.name, lambda v: min_val <= v <= max_val)

    @classmethod
    def do_min_filter(cls, data, filter_feature) -> Any:
        value = filter_feature.parameter.value
        return cls._keep(data, filter_feature.name, lambda v: v >= value)

    @classmethod
    def _apply_max_inclusive_filter(cls, data, column_name, threshold) -> Any:
        return cls._keep(data, column_name, lambda v: v <= threshold)

    @classmethod
    def _apply_max_exclusive_filter(cls, data, column_name, threshold) -> Any:
        return cls._keep(data, column_name, lambda v: v < threshold)

    @classmethod
    def do_equal_filter(cls, data, filter_feature) -> Any:
        value = filter_feature.parameter.value
        return cls._keep(data, filter_feature.name, lambda v: v == value)

    @classmethod
    def do_regex_filter(cls, data, filter_feature) -> Any:
        regex = re.compile(filter_feature.parameter.value)
        return cls._keep(data, filter_feature.name, lambda v: regex.search(str(v)) is not None)

    @classmethod
    def do_categorical_inclusion_filter(cls, data, filter_feature) -> Any:
        values = filter_feature.parameter.values
        keep_missing = any(_is_missing(v) for v in values)
        present = {v for v in values if not _is_missing(v)}
        col = filter_feature.name
        return [row for row in data if row.get(col) in present or (keep_missing and _is_missing(row.get(col)))]
```

`filter_feature.name` is the resolved column name. `FilterEngineTestMixin` runs the shared contract suite against an engine; see the [Testing Guide](09-testing-guide.md).

## Timezone Validation (Opt-In)

Range/min/max filters with a native `datetime` bound (pandas `Timestamp` counts) can be guarded:
bound and column timezone-awareness must match. Opt in via a class attribute, mirroring the
merge-engine flag:

```python
class MyFilterEngine(BaseFilterEngine):
    provides_column_semantics = True

    @classmethod
    def _column_semantics(cls, data, column) -> "ColumnSemantics": ...  # see 06-merge-engine
```

- Default `False`: guard skipped, hook never required.
- Fires only when the bound is a `datetime` and the column is temporal; numeric and string
  filters are unaffected.
- Opted in without the hook: `NotImplementedError`.

Hook contract and example: [06-merge-engine](06-merge-engine.md#timezone-validation-opt-in).
Full model: upstream
[comparison contract](https://github.com/mloda-ai/mloda/blob/main/docs/docs/in_depth/comparison-contract.md).

## FeatureGroup Override

Individual FeatureGroups can override filter behavior per-class by defining `final_filters()` on the FeatureGroup itself. This takes precedence over the FilterEngine's `final_filters()` setting.

| `FeatureGroup.final_filters()` | Effect |
|------|--------|
| `None` (default) | Defer to the FilterEngine's `final_filters()`. |
| `False` | Skip row elimination for this FeatureGroup. |
| `True` | Force row elimination, even if the FilterEngine returns `False`. |

The framework checks `FeatureGroup.final_filters()` first. Only when it returns `None` does it fall back to `FilterEngine.final_filters()`.

See [Filter Concepts](../feature-group-patterns/15-filter-concepts.md) for usage patterns.

## Test

```python
def test_filter_engine():
    data = [{"value": 1}, {"value": 5}, {"value": 10}]
    filter_feature = SingleFilter(
        filter_feature=Feature.int32_of("value"),
        filter_type="min",
        parameter={"value": 5},
    )
    result = MyFilterEngine.do_min_filter(data, filter_feature)
    assert len(result) == 2  # value >= 5
```

## Real Implementations

| File | Description |
|------|-------------|
| [pandas/pandas_filter_engine.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/compute_framework/base_implementations/pandas/pandas_filter_engine.py) | Pandas |
| [polars/polars_filter_engine.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/compute_framework/base_implementations/polars/polars_filter_engine.py) | Polars |
| [python_dict/python_dict_filter_engine.py](https://github.com/mloda-ai/mloda/blob/main/mloda_plugins/compute_framework/base_implementations/python_dict/python_dict_filter_engine.py) | Pure Python |
