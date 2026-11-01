# Pattern 8: Links and Joins

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


# Source groups the joins below refer to; each exposes its join key via index_columns().
class OrderFeatureGroup(FeatureGroup):
    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("order_id",))]


class CustomerFeatureGroup(FeatureGroup):
    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("customer_id",))]


class EventFeatureGroup(FeatureGroup):
    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("symbol",))]


class QuoteFeatureGroup(FeatureGroup):
    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("symbol",))]


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

A link on one input is enough; core injects the join key into the unlinked siblings.

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

## As-of (point-in-time) joins

A plain `Link.inner()` / `Link.left()` is an equi-join: rows pair up only when their
join keys are exactly equal. An as-of join adds a per-row time match on top of that: for
each left row, mloda picks the right row that was valid at the left row's timestamp.
This is the join that recsys, slowly-changing-dimension lookups, and event-vs-state
pipelines need, and it cannot be expressed as an equi-join.

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
| PyArrow | yes | Native Acero `Table.join_asof` ([mloda#489](https://github.com/mloda-ai/mloda/pull/489)). Rejects `direction="nearest"`, `allow_exact_matches=False`, and non-integer tolerances (integer only). |
| DuckDB | yes | SQL `ASOF JOIN`. Rejects `direction="nearest"` and a `timedelta` tolerance with a `ValueError`; pass a numeric tolerance instead. |
| SQLite | yes | SQL window functions. Same restriction as DuckDB: no `nearest`, numeric tolerance only. |

On every backend the as-of time columns must be ordered: a non-ordered (e.g. string) column raises a `ValueError` naming it, unless `coerce_time_columns=True` coerces an ISO-8601 string column.

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
import pandas as pd
from mloda.provider import DataCreatorFeatureGroup
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


class RunEvents(EventFeatureGroup, DataCreatorFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"id", "symbol", "event_ts"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        ts = pd.to_datetime(["2023-01-01 10:15", "2023-01-01 10:45"])
        return pd.DataFrame({"id": [0, 1], "symbol": ["AAA", "AAA"], "event_ts": ts})


class RunQuotes(QuoteFeatureGroup, DataCreatorFeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"symbol", "quote_ts", "price"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        ts = pd.to_datetime(["2023-01-01 10:00", "2023-01-01 10:30"])
        return pd.DataFrame({"symbol": ["AAA", "AAA"], "quote_ts": ts, "price": [100, 110]})


class EventPriceFeatureGroup(FeatureGroup):
    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"asof_event_id", "asof_event_price"}

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("id"), Feature("event_ts"), Feature("quote_ts"), Feature("price")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"asof_event_id": data["id"], "asof_event_price": data["price"]})


link = Link.asof_on(
    RunEvents, RunQuotes, left_time_column="event_ts", right_time_column="quote_ts", direction="backward"
)

results = mloda.run_all(
    [Feature("asof_event_id"), Feature("asof_event_price")],
    compute_frameworks=[PandasDataFrame],
    links={link},
    plugin_collector=PluginCollector.enabled_feature_groups({RunEvents, RunQuotes, EventPriceFeatureGroup}),
)
assert list(results[0]["asof_event_price"]) == [100, 110]
```

A complete, runnable version (two `DataCreatorFeatureGroup`-backed feature groups, a small
hand-traced dataset, and the backward-join semantics asserted row by row) lives in the
registry test suite:
[`tests/test_end2end/test_asof_join_example.py`](https://github.com/mloda-ai/mloda-registry/blob/main/tests/test_end2end/test_asof_join_example.py).

## Using JoinSpec (Explicit Control)

Specifies join columns directly. No `index_columns()` needed on the feature group.
The engine automatically injects the specified join columns into the feature group's feature set.

```py
link = Link.inner(left=JoinSpec(FeatureGroupA, "id"), right=JoinSpec(FeatureGroupB, "ref_id"))
```

## Using _on Methods (Convenience)

Auto-derives join columns from `index_columns()`.

```py
link = Link.inner_on(UserFeatureGroup, OrderFeatureGroup)

# Select specific index position
link = Link.inner_on(UserFG, OrderFG, left_index=0, right_index=1)
```

## Self-Joins with Aliases

```py
link = Link.inner_on(
    UserFeatureGroup, UserFeatureGroup, self_left_alias={"side": "left"}, self_right_alias={"side": "right"}
)

features = {
    Feature("age", options={"side": "left"}),
    Feature("age", options={"side": "right"}),
}
```

## Via Feature Parameter

```python
# Pass link when creating a Feature
link = Link.left(
    JoinSpec(OrderFeatureGroup, Index(("order_id",))), JoinSpec(CustomerFeatureGroup, Index(("customer_id",)))
)
feature = Feature(name="order_value", link=link, index=Index(("order_id",)))
```

## Same-Class Joins with Discriminators

When both sides use the same FeatureGroup class, use discriminators to distinguish them. Two features pointed at different files of one source group (`CsvFG`) are the typical case: the discriminator is the pointer option itself.

```python
from mloda.user import Feature, JoinSpec, Link
from mloda_plugins.feature_group.input_data.file_formats.csv_fg import CsvFG

link = Link.inner(
    JoinSpec(CsvFG, "id"),
    JoinSpec(CsvFG, "id"),
    left_discriminator={"CsvFG": "customers.csv"},
    right_discriminator={"CsvFG": "orders.csv"},
)
features = [
    Feature("name", options={"CsvFG": "customers.csv"}),
    Feature("amount", options={"CsvFG": "orders.csv"}),
]
```

## Polymorphic Link Sides

A `JoinSpec` matches its feature group and its subclasses, so a link declared on a base class also joins the subclasses and injects the side's join key into them; each subclass must provide that column. Matching rules: [Polymorphic Link Matching](https://github.com/mloda-ai/mloda/blob/main/docs/docs/in_depth/join_data.md#polymorphic-link-matching).

A link side can therefore have several members, and a subclass may run on another compute framework (its own `compute_framework_rule()`). Since mloda 0.16.0, a consumer that joins two members of one side on different frameworks is checked at plan time. Core's table [One consumer joining two members of one link side](https://github.com/mloda-ai/mloda/blob/0.18.0/docs/docs/in_depth/join_data.md#polymorphic-link-matching) lists the layouts that plan: the other side on one member's framework with the consumer there too, or the other side as one feature group on a framework no member uses, with the consumer on that framework reading both members directly (inner and outer joins, or a left or right join that keeps the other side). Other layouts fail planning with a `ValueError` that names the consumer and the members and tells you to `read one member of the link side per consumer`.

Remedy: split the consumer into one FeatureGroup per member, each requesting that member and the other side in `input_features()`.

The check has limits:

- It does not cover append and union links.
- Members that share a compute framework pass it, but one base-class link over both is still ambiguous (`There are more than one solution for the join`).
- A planned layout is not always correct: with the other side on one member's framework, a left or right join that keeps the member side can return wrong rows when the members hold different keys, and under `THREADING` planned layouts can fail intermittently with a missing column.

## Real Implementations

| File | Description |
|------|-------------|
| [join_data.md](https://github.com/mloda-ai/mloda/blob/main/docs/docs/in_depth/join_data.md) | In-depth join documentation |

## Combines With

- **Index Features** (Pattern 7): `_on` convenience methods require `index_columns()` defined on joined feature groups; `JoinSpec` works regardless
- **Framework-specific** (Pattern 9): Different merge engines per framework
