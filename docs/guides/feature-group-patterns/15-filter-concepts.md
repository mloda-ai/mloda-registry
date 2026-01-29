# Filter Concepts

How to use filters when computing features.

**What**: Filters narrow down data by removing rows based on conditions.
**When**: Computing features on a subset of data (time windows, categories, ranges).
**Why**: Reduces computation, enables time-based and segment analysis.
**Where**: Derived features get filters applied automatically after calculation. Data sources (DB, API, files) should apply filters during data loading via `features.filters`.
**How**: Pass `GlobalFilter` to `mloda.run_all()`.

## Filter Types

| Type | Parameter |
|------|-----------|
| `equal` | `{"value": x}` |
| `min` | `{"value": x}` |
| `max` | `{"value": x}` |
| `range` | `{"min": x, "max": y, "max_exclusive": bool}` |
| `regex` | `{"value": "pattern"}` |
| `categorical_inclusion` | `{"values": [a, b, c]}` |

---

## Basic Usage

```python
from mloda.user import mlodaAPI, Feature, GlobalFilter

global_filter = GlobalFilter()
global_filter.add_filter("age", "range", {"min": 18, "max": 65})
global_filter.add_filter("region", "categorical_inclusion", {"values": ["EU", "NA"]})

result = mlodaAPI.run_all(
    [Feature.not_typed("my_feature")],
    global_filter=global_filter
)
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

---

## Key Constraint

All features from the same FeatureGroup must have the same filters. Split into separate feature groups if needed.
