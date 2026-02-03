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
| `do_max_filter()` | Filter <= value |
| `do_equal_filter()` | Filter == value |
| `do_regex_filter()` | Filter by regex pattern |
| `do_categorical_inclusion_filter()` | Filter by set membership |
| `final_filters()` | Return True if filters applied at end |

## Complete Example

```python
from typing import Any
from mloda.provider import BaseFilterEngine
import re


class MyFilterEngine(BaseFilterEngine):
    """Filter engine for MyFramework."""

    @classmethod
    def final_filters(cls) -> bool:
        return True

    @classmethod
    def do_range_filter(cls, data, filter_feature) -> Any:
        min_val, max_val, is_exclusive = cls.get_min_max_operator(filter_feature)
        col = filter_feature.filter_feature.name
        if is_exclusive:
            return [row for row in data if min_val <= row.get(col) < max_val]
        return [row for row in data if min_val <= row.get(col) <= max_val]

    @classmethod
    def do_min_filter(cls, data, filter_feature) -> Any:
        value = filter_feature.parameter.value
        col = filter_feature.filter_feature.name
        return [row for row in data if row.get(col) >= value]

    @classmethod
    def do_max_filter(cls, data, filter_feature) -> Any:
        value = filter_feature.parameter.value
        col = filter_feature.filter_feature.name
        return [row for row in data if row.get(col) <= value]

    @classmethod
    def do_equal_filter(cls, data, filter_feature) -> Any:
        value = filter_feature.parameter.value
        col = filter_feature.filter_feature.name
        return [row for row in data if row.get(col) == value]

    @classmethod
    def do_regex_filter(cls, data, filter_feature) -> Any:
        pattern = filter_feature.parameter.value
        col = filter_feature.filter_feature.name
        regex = re.compile(pattern)
        return [row for row in data if regex.match(str(row.get(col, "")))]

    @classmethod
    def do_categorical_inclusion_filter(cls, data, filter_feature) -> Any:
        values = set(filter_feature.parameter.values)
        col = filter_feature.filter_feature.name
        return [row for row in data if row.get(col) in values]
```

## Test

```python
def test_filter_engine():
    data = [{"value": 1}, {"value": 5}, {"value": 10}]
    filter_feature = SingleFilter(
        filter_feature=Feature.int32_of("value"),
        filter_type="min",
        parameter=FilterParameter(value=5),
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
