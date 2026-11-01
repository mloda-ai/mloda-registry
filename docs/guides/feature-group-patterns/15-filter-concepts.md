# Filter Concepts

How to use filters when computing features.

**What**: Filters narrow down data by removing rows based on conditions.
**When**: Computing features on a subset of data (time windows, categories, ranges).
**Why**: Reduces computation, enables time-based and segment analysis.
**Where**: Derived features get filters applied automatically after calculation. Data sources (DB, API, files) should apply filters during data loading via `features.filters`.
**How**: Pass `GlobalFilter` to `mloda.run_all()`.

## Filter Types

| Type | Enum | Parameter |
|------|------|-----------|
| `equal` | `FilterType.EQUAL` | `{"value": x}` |
| `min` | `FilterType.MIN` | `{"value": x}` |
| `max` | `FilterType.MAX` | `{"value": x}` |
| `range` | `FilterType.RANGE` | `{"min": x, "max": y, "max_exclusive": bool}` |
| `regex` | `FilterType.REGEX` | `{"value": "pattern"}` |
| `categorical_inclusion` | `FilterType.CATEGORICAL_INCLUSION` | `{"values": [a, b, c]}` |

---

## Basic Usage

```python
from mloda.user import mloda, Feature, GlobalFilter

global_filter = GlobalFilter()
global_filter.add_filter("age", "range", {"min": 18, "max": 65})
global_filter.add_filter("region", "categorical_inclusion", {"values": ["EU", "NA"]})

api_data = {"Customers": {"age": [17, 30, 70], "region": ["EU", "NA", "EU"], "my_feature": [1, 2, 3]}}
features = [Feature.not_typed(name) for name in ("age", "region", "my_feature")]

result = mloda.run_all(features, global_filter=global_filter, api_data=api_data)
assert result[0]["age"] == [30]
```

### Missing Values and Value Types

- A null or NaN row never passes a range, min, max or equal filter. Categorical inclusion keeps null and NaN rows only when `values` contains `None` (a NaN value counts as `None`).
- `values` must be a list, tuple, set or frozenset; a string raises `TypeError` in `add_filter`.

Every filter engine follows these rules; see [Filter Data](https://mloda-ai.github.io/mloda/in_depth/filter_data/) for the full contract.

---

## Time Filters

```python
from datetime import datetime, timezone

global_filter = GlobalFilter()
global_filter.add_time_and_time_travel_filters(
    event_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
    event_to=datetime(2024, 12, 31, tzinfo=timezone.utc),
    max_exclusive=True,
    event_time_column="reference_time",
)
```

---

## Controlling Row Elimination with `final_filters()`

By default, the framework defers to the `FilterEngine` to decide whether row elimination runs after `calculate_feature()`. A FeatureGroup can override this by defining `final_filters()`:

```python
from mloda.provider import FeatureGroup


class MyDataSource(FeatureGroup):
    @classmethod
    def final_filters(cls) -> bool | None:
        return False  # Skip row elimination; we handle filters inline
```

| Return value | Behavior |
|--------------|----------|
| `None` (default) | Defer to the FilterEngine's `final_filters()` setting. |
| `False` | Skip row elimination. The FeatureGroup handles filters itself (e.g. predicate pushdown). |
| `True` | Force row elimination after calculation, regardless of FilterEngine setting. |

`features.filters` is always available inside `calculate_feature()`, regardless of what `final_filters()` returns. A FeatureGroup can read filters inline for conditional masking or predicate pushdown and still request row elimination by returning `True`.

### Usage Patterns

**Pattern 1: Full inline handling (skip row elimination)**

The FeatureGroup applies all filters during data loading. No post-calculation filtering needed.

```python
class SqlDataSource(FeatureGroup):
    @classmethod
    def final_filters(cls) -> bool | None:
        return False

    @classmethod
    def calculate_feature(cls, data, features):
        query = "SELECT * FROM table WHERE 1=1"
        for f in features.filters:
            if f.filter_type == "equal":
                query += f" AND {f.filter_feature.name} = {f.parameter.value}"
            elif f.filter_type == "range":
                query += f" AND {f.filter_feature.name} BETWEEN {f.parameter.min_value} AND {f.parameter.max_value}"
        # All filtering done in SQL
        return execute(query)
```

**Pattern 2: Force row elimination**

The FeatureGroup reads filters inline for conditional logic but still wants the framework to eliminate rows afterward.

```python
class ConditionalFeature(FeatureGroup):
    @classmethod
    def final_filters(cls) -> bool | None:
        return True  # Force row elimination after calculation

    @classmethod
    def calculate_feature(cls, data, features):
        for f in features.filters:
            if f.filter_type == "equal":
                # Use filter for conditional masking during calculation
                mask = data["category"] == f.parameter.value
                data["score"] = data["score"] * mask
        return data  # Framework will still eliminate non-matching rows
```

---

## The Overlap Contract

When `final_filters()` returns `True` (or the FilterEngine applies row elimination), the framework validates that every filter column exists in the FeatureGroup's output data. If a filter references a column that is not present, the framework raises a `ValueError`.

**What to watch for:**
- If you return `True` from `final_filters()`, make sure filter columns survive your calculation (do not drop them from the output).
- If a filter column is not in your FeatureSet's output names, the filter is silently skipped (it does not apply to your FeatureGroup).

---

## Key Constraint

All features from the same FeatureGroup must have the same filters. Split into separate feature groups if needed.

---

## Related

- [Filter Engine](../compute-framework-patterns/07-filter-engine.md) - Implementing filter operations in a compute framework
- [calculate_feature](12-calculate-feature.md) - Accessing `features.filters` inside the calculation method, and the [empty-result contract](12-calculate-feature.md#empty-results) a filter can trigger
- [Masking](25-masking.md) - Conditional aggregation that nulls out values instead of removing rows
