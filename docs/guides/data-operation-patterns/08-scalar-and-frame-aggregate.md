# Scalar and Frame Aggregate

Two row-preserving aggregate families. Scalar aggregate broadcasts one value across the whole table. Frame aggregate broadcasts a value computed over a bounded window (rolling, time-window, cumulative, or expanding).

**What**: `ScalarAggregateFeatureGroup` and `FrameAggregateFeatureGroup` compute aggregates without reducing rows.
**When**: You need a reference value on every row (scalar) or a moving aggregate (frame).
**Why**: Both preserve row count, which lets them chain with other row-preserving ops. The group-reducing variant lives in `aggregation/` and is covered separately.
**Where**: `mloda/community/feature_groups/data_operations/row_preserving/{scalar_aggregate,frame_aggregate}/`.
**How**: Encode the agg type in the feature name. Frame variants also encode the window kind and size.

---

## Scalar aggregate

Pattern: `{col}__{agg}_scalar` (regex `r".*__([\w]+)_scalar$"`).

Every row receives the same aggregate value. No partitioning, no windowing.

```python
from mloda.user import Feature, PluginLoader, mloda

PluginLoader.all()

# Every row gets the global max
feature = Feature("value_int__max_scalar")
```

Supported aggregations are `sum`, `min`, `max`, `avg`/`mean`, `count`, `std`/`std_pop`/`std_samp`, `var`/`var_pop`/`var_samp`, and `median`. `mode`, `nunique`, `first`, and `last` are not supported on scalar aggregate, only on [window aggregation](06-window-aggregation.md); they need ordering or group structure that a single global scalar does not provide.

Scalar aggregate accepts the `mask` option. Masked rows have their source value replaced with NULL before the aggregate computes; the output still has the same row count.

---

## Frame aggregate

Frame aggregate supports four window kinds. The feature name encodes which kind and its size.

| Kind | Pattern | Example |
|---|---|---|
| Rolling (row-count window) | `{col}__{agg}_rolling_{N}` | `value__sum_rolling_3` |
| Time-based window | `{col}__{agg}_{size}_{unit}_window` | `value__avg_7_day_window` |
| Cumulative | `{col}__cum{agg}` | `value__cumsum`, `value__cummax` |
| Expanding | `{col}__expanding_{agg}` | `value__expanding_avg` |

### Rolling

`value__sum_rolling_3` computes the sum of the current row plus the two preceding, giving a 3-row rolling sum. The first `N-1` rows per partition return NULL (incomplete window).

```python
Feature(
    "value__sum_rolling_3",
    Options(context={"partition_by": ["customer_id"], "order_by": "ts"}),
)
```

### Time window

`value__avg_7_day_window` computes an average over all rows whose `order_by` timestamp falls within the last 7 days of the current row's timestamp. Supported units include `day`, `hour`, `minute`, `second`. The `order_by` column must be a timestamp.

### Cumulative

`value__cumsum` is the running sum from the start of the partition to the current row. Cumulative variants: `cumsum`, `cummin`, `cummax`, `cumcount`.

### Expanding

`value__expanding_avg` is the aggregate of all rows from the start of the partition up to and including the current row. Any supported aggregation works.

---

## Shared options

All frame-aggregate variants accept:

| Key | Type | Purpose |
|---|---|---|
| `partition_by` | `list[str]` | Resets the window at each partition boundary |
| `order_by` | `str` | Required for rolling, time-window, cumulative, and expanding |
| `mask` | tuple or list of tuples | Conditional aggregation; see [Masking](../feature-group-patterns/25-masking.md) |

---

## Scalar vs frame vs window vs aggregation

Four nearby concepts; easy to mix up.

| Feature group | Pattern suffix | Row behavior | Scope |
|---|---|---|---|
| Scalar aggregate | `_scalar` | Preserves (broadcasts one value globally) | Whole table |
| Window aggregation | `_window` | Preserves (broadcasts per partition) | Partition |
| Frame aggregate | `_rolling_N`, `_{size}_{unit}_window`, `cum*`, `expanding_*` | Preserves (broadcasts per bounded window) | Row-relative window |
| Aggregation | `_agg` | Reduces to one row per group | Partition |

Pick by what your downstream step needs: same row count and a reference value (scalar), same row count and per-partition context (window), same row count with a moving window (frame), or a collapsed group-by table (aggregation).

---

## Related

- [Window aggregation](06-window-aggregation.md) - Per-partition aggregates without the rolling/expanding dimension.
- [Row-preserving contract](02-row-preserving-contract.md) - Why the output row count matches the input for all frame kinds.
- [Masking](../feature-group-patterns/25-masking.md) - Conditional aggregation for all of these variants.
