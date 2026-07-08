# Resample

Resample collapses an irregular event stream onto a regular time grid: it floors each row's timestamp to a fixed bucket, groups by that bucket (and any partitions), and aggregates. Unlike every other data operation, it **changes the row count**.

**What**: `ResampleFeatureGroup` accepts feature names of the form `{col}__resample_{n}_{unit}_{agg}` (e.g. `value__resample_1_hour_mean`). It reads `time_column` (required) and optional `partition_by` from `Options(context=...)` and emits one row per non-empty bucket.
**When**: You want fixed-interval features ("hourly mean", "15-minute sum", "daily count") from timestamped events.
**Why**: Regridding consistently across frameworks requires pinning the bucket anchor, the empty-bucket policy, and the null-aggregation behavior; these all diverge by default.
**Where**: `mloda/community/feature_groups/data_operations/row_changing/resample/`.
**How**: The feature name carries the bucket size, unit, and aggregation; the time column and partitions come from Options.

---

## A new category: `row_changing`

Resample is the first member of the `row_changing/` category. Existing categories either preserve the row count (`row_preserving/`, `string/`) or reduce to one row per group (`aggregation/`). Resample is neither: it collapses events onto a regular grid, producing one row per occupied time bucket per partition. The standard row-by-row cross-framework comparison does not apply; resample is verified with a result-map comparison (keyed by partition + bucket) against the PyArrow reference, mirroring how `aggregation` is tested.

---

## Pattern

```text
value__resample_1_hour_mean
value__resample_15_minute_sum
value__resample_1_day_count
```

| Segment | Domain |
|---|---|
| `n` | Any positive integer bucket size. |
| `unit` | `minute`, `hour`, `day` (fixed-frequency only in v1). |
| `agg` | `mean`, `sum`, `count`, `min`, `max` (order-independent only in v1). |

| Option | Required | Meaning |
|---|---|---|
| `time_column` | yes | The timestamp column to bucket on. |
| `partition_by` | no (default `[]`) | Extra group keys alongside the bucket. |
| `in_features` | config form only | The source value column to aggregate. |

`first` / `last` and calendar units (`week` / `month` / `year`) are intentionally out of scope for v1: the former reintroduce intra-bucket tie-break ambiguity, the latter need non-uniform bucket widths.

---

## Semantics

- **Bucket anchor.** The `time_column` is floored to its `n * unit` bucket using the same epoch-anchored fixed-frequency floor as [time bucketization](11-time-bucketization.md), so buckets align identically across backends.
- **One row per non-empty bucket.** Empty (gap) buckets are not emitted. The pandas backend groups by the floored timestamp rather than using `.resample()`, which would otherwise materialize gap rows that the group-by backends omit.
- **Output columns.** The `partition_by` columns, the bucketed `time_column` (same name, holding the bucket start), and the aggregated column named exactly `{col}__resample_{n}_{unit}_{agg}`. Output row order is not guaranteed.
- **Null handling (pinned to the PyArrow oracle).** `mean` / `sum` / `min` / `max` skip nulls; `count` counts non-null values. A bucket that has rows but whose values are all null still emits, with `count = 0` and `mean` / `sum = None`.

The all-null `sum` cell is the one place backends disagree by default (pandas → `0.0`, PyArrow → `None`); the implementation forces every backend to the PyArrow `None` (pandas `min_count=1`, polars `when(count > 0)` guard, DuckDB/SQL `SUM` returns `NULL` natively).

---

## Backend support

| Backend | Mechanism |
|---|---|
| PyArrow (reference) | `floor_temporal` → `group_by([*partition, bucket]).aggregate(...)`. |
| Pandas | `dt.floor` → `groupby([*partition, bucket], dropna=False).agg(...)` (`sum` with `min_count=1`). |
| Polars (lazy) | floor via `dt.truncate` → `group_by(maintain_order=True)` with an all-null `sum` guard. |
| DuckDB | epoch-anchored floor (not the native `time_bucket` 2000-01-03 anchor) → `GROUP BY`. |
| SQLite | not implemented in v1. |

---

## Usage

```python
from mloda.user import Feature, Options, PluginLoader, mloda

PluginLoader.all()  # cached & shared; force_reload=True to pick up newly installed plugins

features = [
    Feature("price__resample_1_hour_mean", Options(context={"time_column": "ts", "partition_by": ["symbol"]})),
]
result = mloda.run_all(features, compute_frameworks={"PandasDataFrame"})
```

The result has one row per occupied hour per symbol, not one row per input event.

---

## Related

- [Time bucketization](11-time-bucketization.md) - The row-preserving floor that resample reuses for bucket alignment.
- [Adding a new data operation](10-adding-new-operation.md) - Step 1 covers choosing a top-level category for a row-count-changing op.
- [Reference implementation pattern](03-reference-implementation.md) - PyArrow is the resample oracle.
