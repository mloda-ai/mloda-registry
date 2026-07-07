# Pattern 8: Links and Joins

Join feature groups to combine data from different sources.

**What**: Features that join data from multiple feature groups.
**When**: Combining data from different sources, cross-source feature combinations, self-joins.
**Why**: Real-world features often require data from multiple tables/sources.
**Where**: Order + Customer joins, user + activity joins, enrichment lookups.
**How**: Attach `Link` to features via `input_features()`, Feature parameter, or `mloda.run_all()`.

## Key Characteristic

| Aspect | Value |
|--------|-------|
| Class | `Link` for defining joins |
| Class | `JoinSpec` for specifying join sides |
| Usage | Attach to Feature objects or pass to `mloda.run_all(links=...)` |

## Complete Example

```python
from typing import Any
from mloda.user import Link, JoinSpec, Index, Feature, FeatureName, Options
from mloda.provider import FeatureGroup, FeatureSet


class OrderWithCustomer(FeatureGroup):
    """Join orders with customer data."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        link = Link.inner_on(OrderFeatureGroup, CustomerFeatureGroup)
        return {
            Feature(name="order_value", link=link),
            Feature(name="customer_name"),
        }

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Data is already joined by the framework
        return data
```

## Test

```python
def test_input_features_with_link():
    fg = OrderWithCustomer()
    features = fg.input_features(Options({}), FeatureName("test"))
    assert features is not None
    assert any(f.link is not None for f in features)
```

## Join Types

| Type | SQL Equivalent | Method |
|------|----------------|--------|
| `Link.inner()` | INNER JOIN | `inner_on()` |
| `Link.left()` | LEFT JOIN | `left_on()` |
| `Link.right()` | RIGHT JOIN | `right_on()` |
| `Link.outer()` | FULL OUTER JOIN | `outer_on()` |
| `Link.append()` | UNION ALL | `append_on()` |
| `Link.union()` | UNION | `union_on()` |
| `Link.asof()` | ASOF JOIN (point-in-time) | `asof_on()` |

ASOF (point-in-time) joins match each left row to the right row that was valid at
the left row's timestamp. See [As-of (point-in-time) joins](#as-of-point-in-time-joins)
below for the keyword reference, per-backend support, and a runnable end-to-end example.

## As-of (point-in-time) joins

A plain `Link.inner()` / `Link.left()` is an equi-join: rows pair up only when their
join keys are exactly equal. An as-of join (added in mloda 0.8.0) adds a per-row time
match on top of that: for each left row, mloda picks the right row that was valid at
the left row's timestamp. This is the join that recsys, slowly-changing-dimension
lookups, and event-vs-state pipelines need, and it cannot be expressed as an equi-join.

Two matches happen at once:

- **Equi match on the by-key.** The join `Index` (derived from `index_columns()` when
  you use `asof_on`, or given explicitly via `JoinSpec` when you use `asof`) is matched
  for equality, exactly like an inner join. Rows only pair up within the same key.
- **Inequality match on time.** Within each key, the left row's `left_time_column` is
  matched against the right row's `right_time_column` according to `direction`.

| Keyword | Default | Meaning |
|---|---|---|
| `left_time_column` | required | Time column on the left side. |
| `right_time_column` | required | Time column on the right side. |
| `direction` | `"backward"` | `"backward"`: latest right row at or before the left time. `"forward"`: earliest right row at or after it. `"nearest"`: whichever is closest in time. |
| `tolerance` | `None` | Maximum allowed time gap; a left row with no right match inside the gap gets nulls. Accepts a number or a `timedelta` (backend restrictions apply, see below). |
| `allow_exact_matches` | `True` | Whether an exactly-equal timestamp counts as a match. |
| `coerce_time_columns` | `False` | When `True`, ISO-8601 string time columns are coerced to native timestamps per backend instead of raising. The default keeps the strict ordered-dtype check. |

These follow pandas `merge_asof` semantics. The keyword arguments are keyword-only on
both factories (`Link.asof(left_spec, right_spec, *, left_time_column=..., ...)` and
`Link.asof_on(LeftFG, RightFG, *, left_time_column=..., ...)`).

### Backend support

| Backend | As-of | Notes |
|---|---|---|
| Pandas | yes | Native `pd.merge_asof`. Supports `direction="nearest"` and `timedelta` tolerances. |
| Polars (lazy) | yes | Native `join_asof`. Supports `direction="nearest"` and `timedelta` tolerances. |
| PyArrow | yes | Native Acero `Table.join_asof` (mloda 0.9.0, [mloda#489](https://github.com/mloda-ai/mloda/pull/489): no more pandas round-trip). Like the SQL backends it rejects `direction="nearest"` with a `ValueError`; it is stricter than they are in two ways, also rejecting `allow_exact_matches=False` (Acero's match range always includes exact matches) and requiring an integer tolerance (a `timedelta` or non-integer tolerance is rejected). |
| DuckDB | yes | SQL `ASOF JOIN`. Rejects `direction="nearest"` and a `timedelta` tolerance with a `ValueError`; pass a numeric tolerance instead. |
| SQLite | yes | SQL window functions. Same restriction as DuckDB: no `nearest`, numeric tolerance only. |

`direction="nearest"` and `timedelta` tolerances work on the Pandas and Polars
backends but are rejected up front (with a `ValueError`) on the PyArrow, DuckDB, and
SQLite backends rather than emulated in Python: PyArrow's native Acero join has no
symmetric "nearest" mode and takes an integer tolerance, and the SQL backends want a
numeric tolerance. Pick a backend that supports the knobs your join needs.

Two guards apply uniformly across every backend (mloda 0.9.0):

- **Ordered time columns ([mloda#529](https://github.com/mloda-ai/mloda/pull/529)).**
  As-of time columns must be ordered (datetime, numeric, or timedelta). A non-ordered
  column (for example ISO-8601 date *strings* / object dtype) raises a clear
  `ValueError` naming the column instead of silently producing wrong matches. Cast it
  to a real datetime or numeric before joining.
- **Opt-in string coercion ([mloda#548](https://github.com/mloda-ai/mloda/pull/548)).**
  Passing `coerce_time_columns=True` to `Link.asof(...)` / `Link.asof_on(...)` opts in
  to coercing ISO-8601 string time columns to native timestamps per backend; the
  default (`False`) keeps the strict error above.

### Defining the link

With `asof_on`, both sides expose the by-key through `index_columns()`:

```python
from mloda.user import Link

link = Link.asof_on(
    EventFeatureGroup,
    QuoteFeatureGroup,
    left_time_column="event_ts",
    right_time_column="quote_ts",
    direction="backward",
)
```

When a side has no `index_columns()`, name the by-key explicitly with `JoinSpec`:

```python
from mloda.user import Index, JoinSpec, Link

link = Link.asof(
    JoinSpec(EventFeatureGroup, Index(("symbol",))),
    JoinSpec(QuoteFeatureGroup, Index(("symbol",))),
    left_time_column="event_ts",
    right_time_column="quote_ts",
    direction="backward",
)
```

### Running it

The join fires when a consumer feature group requests features that resolve to both
linked sides; mloda hands the merged frame to that group's `calculate_feature`. Pass
the link to `run_all` (the `links` parameter is a `set`):

```python
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

results = mloda.run_all(
    [Feature("asof_event_id"), Feature("asof_event_price")],
    compute_frameworks={PandasDataFrame},
    links={link},
    plugin_collector=PluginCollector.enabled_feature_groups(
        {EventFeatureGroup, QuoteFeatureGroup, EventPriceFeatureGroup}
    ),
)
```

A complete, runnable version (two `DataCreator`-backed source feature groups, a small
hand-traced dataset, and the backward-join semantics asserted row by row) lives in the
registry test suite:
[`tests/test_end2end/test_asof_join_example.py`](https://github.com/mloda-ai/mloda-registry/blob/main/tests/test_end2end/test_asof_join_example.py).

## Using JoinSpec (Explicit Control)

Specifies join columns directly. No `index_columns()` needed on the feature group.
The engine automatically injects the specified join columns into the feature group's feature set.

```python
link = Link.inner(
    left=JoinSpec(FeatureGroupA, "id"),
    right=JoinSpec(FeatureGroupB, "ref_id")
)
```

## Using _on Methods (Convenience)

Auto-derives join columns from `index_columns()`.

```python
link = Link.inner_on(UserFeatureGroup, OrderFeatureGroup)

# Select specific index position
link = Link.inner_on(UserFG, OrderFG, left_index=0, right_index=1)
```

## Self-Joins with Aliases

```python
link = Link.inner_on(UserFeatureGroup, UserFeatureGroup,
                     self_left_alias={"side": "left"},
                     self_right_alias={"side": "right"})

features = {
    Feature("age", options={"side": "left"}),
    Feature("age", options={"side": "right"}),
}
```

## Feature-Level Links

Links can also be set directly on Feature objects.

## Via Feature Parameter

```python
# Pass link when creating a Feature
link = Link.left(
    JoinSpec(OrderFeatureGroup, Index(("order_id",))),
    JoinSpec(CustomerFeatureGroup, Index(("customer_id",)))
)
feature = Feature(name="order_value", link=link, index=Index(("order_id",)))
```

## Via input_features()

```python
def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
    link = Link.left(
        JoinSpec(OrderFeatureGroup, Index(("order_id",))),
        JoinSpec(CustomerFeatureGroup, Index(("customer_id",)))
    )
    return {
        Feature(name="order_value", link=link, index=Index(("order_id",))),
        Feature(name="customer_name", index=Index(("customer_id",))),
    }
```

## Multi-Table Join (Aggregate + Left-Join)

```python
link = Link.left(
    left=JoinSpec(CustomerFeatures, Index(("customer_id",))),
    right=JoinSpec(OrderAggregation, Index(("customer_id",))),
)

features = [
    Feature("customer_name"),
    Feature("total_orders", link=link),
]
```


## Same-Class Joins with Discriminators

When both sides use the same FeatureGroup class, use discriminators to distinguish them:

```python
link = Link.inner(
    JoinSpec(ReadFileFeature, "id"),
    JoinSpec(ReadFileFeature, "id"),
    left_discriminator={"CsvReader": "customers.csv"},
    right_discriminator={"CsvReader": "orders.csv"},
)
```

## When to Use Each Approach

| Approach | Use When |
|----------|----------|
| `input_features()` with link | Joins for derived feature dependencies |
| Feature `link` parameter | Dynamic joins at feature creation |
| `mloda.run_all(links=...)` | Global joins for entire computation |

## Real Implementations

| File | Description |
|------|-------------|
| [join_data.md](https://github.com/mloda-ai/mloda/blob/main/docs/docs/in_depth/join_data.md) | In-depth join documentation |

## Combines With

- **Index Features** (Pattern 7): `_on` convenience methods require `index_columns()` defined on joined feature groups; `JoinSpec` works regardless
- **Framework-specific** (Pattern 9): Different merge engines per framework
