# Window Aggregation

Window aggregation computes a group aggregate (sum, avg, etc.) and broadcasts it back onto every row. Row count is preserved; every row receives the aggregate value for its partition.

**What**: `WindowAggregationFeatureGroup` handles feature names of the form `{col}__{agg}_window`.
**When**: You want a group aggregate alongside the original rows (e.g. "total sales per region on every order row").
**Why**: Unlike `aggregation`, which reduces rows, window aggregation keeps them. You get per-row context plus a per-partition aggregate without a join.
**Where**: `mloda/community/feature_groups/data_operations/row_preserving/window_aggregation/`.
**How**: Encode the aggregation in the feature name. Pass `partition_by` (and optionally `order_by` for ordered aggregates) via `Options(context=...)`.

---

## Supported aggregation types

The matcher accepts any token that falls inside the name regex `r".*__([\w]+)_window$"` and is also present in the supported set:

```
sum, avg, mean, count, min, max,
std, var, std_pop, std_samp, var_pop, var_samp,
median, mode, nunique, first, last
```

`avg` and `mean` are aliases. `std`/`std_pop` and `var`/`var_pop` are aliases (ddof=0). `std_samp` and `var_samp` use ddof=1.

---

## Parameters via `Options(context=...)`

| Key | Type | Required | Purpose |
|---|---|---|---|
| `partition_by` | `list[str]` | Optional | Columns defining the window. Omit for a global window. |
| `order_by` | `str` | Optional | Column that orders rows within each partition. Some agg types (`first`, `last`) rely on it. |
| `mask` | tuple or list of tuples | Optional | Conditional aggregation. See [Masking](../feature-group-patterns/25-masking.md). |

---

## Usage

```python
from mloda.user import Feature, Options, PluginLoader, mloda

PluginLoader.all()

features = [
    Feature(
        "value_int__sum_window",
        Options(context={"partition_by": ["region"]}),
    ),
    Feature(
        "value_int__first_window",
        Options(context={"partition_by": ["region"], "order_by": "ts"}),
    ),
]

result = mloda.run_all(features, compute_frameworks={"DuckdbRelation"})
```

Every row in `result` has the same number of columns plus the two new ones. Within each `region`, `value_int__sum_window` holds the same value on every row.

---

## Row-preserving behavior

Because window aggregation must preserve input order (see [row-preserving contract](02-row-preserving-contract.md)), framework implementations rely on native windowed expressions that do not reorder:

| Framework | Native construct |
|---|---|
| PyArrow | `groupby_aggregate` + broadcast back via index mapping |
| Pandas | `groupby(partition_by)[col].transform(agg)` |
| Polars lazy | `col.{agg}().over(partition_by)` |
| DuckDB / SQLite | `{AGG}(col) OVER (PARTITION BY ...)` (no `ORDER BY` when not needed) |

The DuckDB/SQLite form omits `ORDER BY` for non-ordered aggregates (`sum`, `avg`, etc.). Adding one would reorder rows and break the contract for engines that otherwise would not reorder.

---

## Mask support

Every window-aggregation feature accepts the `mask` option. Masked rows have their source value replaced with NULL before the aggregate runs, so the aggregate skips them. Row count is still preserved; the masked rows remain in the output, just carrying a non-contributing NULL.

```python
Feature(
    "value_int__sum_window",
    Options(context={
        "partition_by": ["region"],
        "mask": ("category", "equal", "X"),
    }),
)
```

See [Masking](../feature-group-patterns/25-masking.md) for the full mask spec.

---

## Related

- [Row-preserving contract](02-row-preserving-contract.md) - Why ordered window functions need careful handling.
- [Scalar and frame aggregate](08-scalar-and-frame-aggregate.md) - Global broadcast and rolling variants.
- [Aggregation naming](../feature-group-patterns/13-feature-naming.md) - How the engine matches `__{agg}_window` suffixes.
