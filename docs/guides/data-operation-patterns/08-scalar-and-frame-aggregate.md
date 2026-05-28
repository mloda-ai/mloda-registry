# Scalar Aggregate, Frame Aggregate, and Scalar Arithmetic

Three row-preserving single-column families. Scalar aggregate broadcasts one value across the whole table. Frame aggregate broadcasts a value computed over a bounded window (rolling, time-window, cumulative, or expanding). Scalar arithmetic combines each value with a numeric constant element-wise.

**What**: `ScalarAggregateFeatureGroup`, `FrameAggregateFeatureGroup`, and `ScalarArithmeticFeatureGroup` compute single-column transforms without reducing rows.
**When**: You need a reference value on every row (scalar), a moving aggregate (frame), or a per-row arithmetic combination with a constant (scalar arithmetic).
**Why**: All three preserve row count, which lets them chain with other row-preserving ops. The group-reducing variant lives in `aggregation/` and is covered separately.
**Where**: `mloda/community/feature_groups/data_operations/row_preserving/{scalar_aggregate,frame_aggregate,scalar_arithmetic}/`.
**How**: Encode the operation in the feature name. Aggregate variants also encode the window kind and size; scalar arithmetic carries the constant in `Options(context={"constant": <number>})`.

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

## Scalar arithmetic

Pattern: `{col}__{op}_constant` (regex `r"(.+?)__(add|subtract|multiply|divide)_constant$"`).

Each row gets `source {op} constant`. Null in the source propagates to the result. Single source column only (`MIN_IN_FEATURES = MAX_IN_FEATURES = 1`).

```python
from mloda.user import Feature, Options, PluginLoader, mloda

PluginLoader.all()

# value_int / 2 element-wise, null preserved
feature = Feature("value_int__divide_constant", Options(context={"constant": 2}))
```

Supported operations are `add`, `subtract`, `multiply`, and `divide`. The constant is carried in `Options(context={"constant": <number>})` and must be `int` or `float` (`bool` is rejected explicitly because it is an `int` subclass that almost certainly indicates a user mistake). `divide` rejects `constant == 0` at validation, before dispatch to any backend.

The source column must be numeric; passing a string or boolean column raises `ValueError` at validation.

`constant` carries `strict_validation=False` on its `PROPERTY_MAPPING` entry so that pattern-only matches (`value_int__add_constant`) succeed without it; the missing-constant check then fires at compute time with a clear error. This lets feature names compose into chains before the constant is bound.

Type semantics across backends:

- `add`, `subtract`, `multiply`: source dtype is preserved. `int + int` stays int on all five backends.
- `divide`: result is always `float64`. PyArrow casts the source via `pc.cast`, DuckDB and SQLite wrap the source in `CAST(... AS DOUBLE/REAL)`, and Pandas / Polars rely on Python's `/` semantics. This is the [cross-backend type-drift mitigation](known-divergences.md) for integer division.

Scalar arithmetic does **not** accept the `mask` option in this initial cut; the source column is passed through unfiltered.

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

`value__avg_7_day_window` computes an average over all rows whose `order_by` timestamp falls within the last 7 days of the current row's timestamp. Units `second`, `minute`, `hour`, `day`, `week` are supported on every framework. `month` and `year` use calendar arithmetic (a 1-month window from Mar 31 reaches back to Feb 28, not 30 days) and are only available on Polars-lazy and DuckDB; Pandas and SQLite reject those units at match time because neither engine has a native calendar-anchored window primitive that matches the reference. See [Known divergences: SQLite + Pandas reject month/year time windows](known-divergences.md#sqlite--pandas-reject-monthyear-time-windows) and the [framework support matrix](framework-support-matrix.md). The `order_by` column must be a timestamp.

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

## Scalar vs scalar arithmetic vs frame vs window vs aggregation

Five nearby concepts; easy to mix up.

| Feature group | Pattern suffix | Row behavior | Scope |
|---|---|---|---|
| Scalar aggregate | `_scalar` | Preserves (broadcasts one value globally) | Whole table |
| Scalar arithmetic | `_constant` | Preserves (per-row source `op` constant) | Each row independently |
| Window aggregation | `_window` | Preserves (broadcasts per partition) | Partition |
| Frame aggregate | `_rolling_N`, `_{size}_{unit}_window`, `cum*`, `expanding_*` | Preserves (broadcasts per bounded window) | Row-relative window |
| Aggregation | `_agg` | Reduces to one row per group | Partition |

Pick by what your downstream step needs: same row count and a reference value (scalar), same row count with an element-wise arithmetic adjustment by a constant (scalar arithmetic), same row count and per-partition context (window), same row count with a moving window (frame), or a collapsed group-by table (aggregation).

---

## Related

- [Window aggregation](06-window-aggregation.md) - Per-partition aggregates without the rolling/expanding dimension.
- [Row-preserving contract](02-row-preserving-contract.md) - Why the output row count matches the input for all frame kinds.
- [Masking](../feature-group-patterns/25-masking.md) - Conditional aggregation for all of these variants.
