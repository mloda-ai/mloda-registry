# Forward fill by time (ffill)

Forward-fill carries the last observed value of a column forward across rows that are null, within a partition, in time order. It is the canonical "last observation carried forward" (LOCF) transform.

**What**: `FfillFeatureGroup` accepts feature names of the form `{col}__ffill`. It reads `order_by` (the time column to sort by, ascending) and optional `partition_by` from `Options(context=...)`, and emits a new `{col}__ffill` column.
**When**: You have an irregular event stream with gaps (sensor readings, prices, status flags) and want every row to see the most recent known value rather than a null.
**Why**: Doing this identically across PyArrow, Pandas, Polars, DuckDB, and SQLite is subtle: partition boundaries, leading-null handling, and ordering stability all diverge by default.
**Where**: `mloda/community/feature_groups/data_operations/row_preserving/ffill/`.
**How**: The feature name carries only the op; the time column and partitions come from Options. Each backend sorts within partition, carries the last non-null forward, and restores the original row order.

---

## Pattern

```text
value__ffill
```

| Option | Required | Meaning |
|---|---|---|
| `order_by` | yes | The time column. Rows are sorted ascending within each partition before the fill. |
| `partition_by` | no (default `[]`) | Columns that scope the fill. With no partition the whole table is one group. |
| `in_features` | config form only | The source column (when not using the `{col}__ffill` string form). |

```python
from mloda.user import Feature, Options, PluginLoader, mloda

PluginLoader.all()

features = [
    Feature("price__ffill", Options(context={"order_by": "ts", "partition_by": ["symbol"]})),
]
result = mloda.run_all(features, compute_frameworks={"PandasDataFrame"})
```

---

## Semantics

- **Row-preserving.** Output has the same rows in the same original order, with one new `{col}__ffill` column appended. The source column is left unchanged.
- **Per partition.** The fill never crosses a `partition_by` boundary. A null at the start of partition B is not filled from partition A's last value.
- **Leading nulls stay null.** A null that precedes the first non-null value in time order has nothing to carry forward, so it remains null.
- **Interior and trailing nulls are filled** from the most recent non-null in time order.

---

## Cross-framework notes

| Backend | Mechanism |
|---|---|
| PyArrow (reference) | `pc.fill_null_forward` applied **per partition slice** after a stable sort. Raw `fill_null_forward` ignores partition boundaries, so the reference fills each group independently and scatters back to original positions. |
| Pandas | sort within partition, `groupby(partition_by, dropna=False)[col].ffill()`, then restore original order via a row-index tag. |
| Polars (lazy) | `pl.col(col).forward_fill().over(partition_by)` after ordering; original order restored from a row index. |
| DuckDB | `LAST_VALUE(col IGNORE NULLS)` over a `partition_by` / `order_by` window (unbounded preceding → current row), via the typed window helper. |
| SQLite | No `IGNORE NULLS`. Uses the two-window fill-group idiom: a running `COUNT(col)` assigns each row to a fill group (leading nulls get group 0), then `MAX(col)` carries the single non-null per group; group 0 stays null. |

Ordering ties matter only when two rows in a partition share an `order_by` value; keep the time column unique within a partition for deterministic results across the SQL backends.

---

## Related

- [Row-preserving contract](02-row-preserving-contract.md) - Output row count and order must match input.
- [Reference implementation pattern](03-reference-implementation.md) - PyArrow's per-partition `fill_null_forward` is the reference.
- [EMA](13-ema.md) - The other ordered, partitioned row-preserving time-series op.
