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

> **Note:** `FilterType` enum members use UPPER_CASE names (`FilterType.EQUAL`, `FilterType.MIN`, etc.). The underlying string values are unchanged, so `add_filter("col", "range", ...)` still works. Using the enum is preferred for type safety.

---

## Basic Usage

```python
from mloda.user import mloda, Feature, GlobalFilter

global_filter = GlobalFilter()
global_filter.add_filter("age", "range", {"min": 18, "max": 65})
global_filter.add_filter("region", "categorical_inclusion", {"values": ["EU", "NA"]})

result = mloda.run_all(
    [Feature.not_typed("my_feature")],
    global_filter=global_filter
)
```

You can also use the `FilterType` enum directly:

```python
from mloda.user import GlobalFilter, FilterType

global_filter = GlobalFilter()
global_filter.add_filter("age", FilterType.RANGE, {"min": 18, "max": 65})
global_filter.add_filter("status", FilterType.EQUAL, {"value": "active"})
```

---

## Time Filters

```python
from datetime import datetime, timezone

global_filter = GlobalFilter()
global_filter.add_time_and_time_travel_filters(
    event_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
    event_to=datetime(2024, 12, 31, tzinfo=timezone.utc),
    max_exclusive=True,
    event_time_column="reference_time"
)
```

---

## Applying Filters in Data Sources

Data sources (DB, API, files) should apply filters during loading. Access via `features.filters`:

```python
@classmethod
def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
    query = "SELECT * FROM table WHERE 1=1"
    for f in features.filters:
        if f.filter_type == "equal":
            query += f" AND {f.filter_feature.name} = {f.parameter.value}"
        elif f.filter_type == "range":
            query += f" AND {f.filter_feature.name} BETWEEN {f.parameter.min_value} AND {f.parameter.max_value}"
    # Execute query...
```

When a data source reads filters inline (e.g. pushing predicates into a SQL WHERE clause), the framework may still attempt row elimination after calculation. Use `final_filters()` to control this behavior (see below).

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
        # All filtering done in SQL
        return execute(query)
```

**Pattern 2: Defer to framework (default)**

The FeatureGroup does not override `final_filters()`. The FilterEngine decides whether to apply row elimination.

```python
class DerivedFeature(FeatureGroup):
    # No final_filters() override, defaults to None (defer to FilterEngine)

    @classmethod
    def calculate_feature(cls, data, features):
        for feature in features.features:
            data[feature.name] = data["source"] * 2
        return data
```

**Pattern 3: Force row elimination**

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

This means: if your FeatureGroup reads filters inline and also requests row elimination, the filter columns must be present in the output. The framework uses `BaseFilterEngine.applicable_filters()` to determine which filters apply (only those whose column name appears in the FeatureSet output), then `_validate_filter_columns()` confirms those columns exist in the actual data before applying them.

**What to watch for:**
- If you return `True` from `final_filters()`, make sure filter columns survive your calculation (do not drop them from the output).
- If a filter column is not in your FeatureSet's output names, the filter is silently skipped (it does not apply to your FeatureGroup).
- The validation protects against bugs where a FeatureGroup claims to produce a column but actually does not include it in the returned data.

---

## Key Constraint

All features from the same FeatureGroup must have the same filters. Split into separate feature groups if needed.

---

## Related

- [Filter Engine](../compute-framework-patterns/07-filter-engine.md) - Implementing filter operations in a compute framework
- [calculate_feature](12-calculate-feature.md) - Accessing `features.filters` inside the calculation method
