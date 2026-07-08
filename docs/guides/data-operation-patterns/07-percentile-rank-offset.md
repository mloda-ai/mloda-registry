# Percentile, Rank, Offset

Three related row-preserving families for analytic windows: percentile (value-at-rank), rank (rank-of-value), and offset (shift-relative-to-row). They share naming and parameter conventions.

**What**: Three feature groups that each compute a per-row analytic value within an optional partition.
**When**: You need percentiles, ranks, lags, or cumulative positions alongside the original rows.
**Why**: These are the core analytic-window primitives that power feature engineering for time series and grouped analytics.
**Where**: `mloda/community/feature_groups/data_operations/row_preserving/{percentile,rank,offset}/`.
**How**: Encode the op and its parameter N into the feature name. Pass `partition_by` (and `order_by` where relevant) via `Options(context=...)`.

---

## Percentile

`PercentileFeatureGroup` computes the value at percentile N within each partition and broadcasts it to every row.

| Pattern | Example |
|---|---|
| `{col}__p{N}_percentile` where N is 0..100 | `latency__p95_percentile` |

```python
from mloda.user import Feature, Options, PluginLoader, mloda

PluginLoader.all()

feature = Feature(
    "latency__p95_percentile",
    Options(context={"partition_by": ["service"]}),
)
```

All rows of the same `service` share the same P95 value. Percentiles use the framework's native method (linear interpolation by default in Pandas and PyArrow). Cross-framework tests allow `pytest.approx` tolerance for floating-point comparisons.

---

## Rank

`RankFeatureGroup` assigns each row a rank within its partition.

| Pattern | Example |
|---|---|
| `{col}__{rank_type}_ranked` | `score__dense_rank_ranked` |

Rank types:

| Type | Semantics |
|---|---|
| `row_number` | Unique 1-based position, ties broken by `order_by` |
| `rank` | Standard rank with gaps (1, 2, 2, 4) |
| `dense_rank` | Standard rank without gaps (1, 2, 2, 3) |
| `percent_rank` | `(rank - 1) / (n - 1)` in `[0, 1]` |
| `ntile_N` | Split into N buckets, return 1..N |
| `top_N` | Boolean: row is in top N by `order_by` |
| `bottom_N` | Boolean: row is in bottom N by `order_by` |

```python
Feature(
    "score__dense_rank_ranked",
    Options(context={"partition_by": ["region"], "order_by": "score"}),
)
```

`order_by` is required for every rank type; the matcher rejects the feature if it is missing. That includes `ntile_N` as well as the standard numeric and boolean variants. Framework test classes may restrict `supported_rank_types` when their engine does not implement one variant; see [Supported ops](04-supported-ops.md).

---

## Offset

`OffsetFeatureGroup` shifts a column by a fixed number of rows inside each partition.

| Pattern | Example |
|---|---|
| `{col}__{offset_type}_offset` | `value__lag_1_offset` |

Offset types:

| Type | Semantics |
|---|---|
| `lag_N` | Value from N rows earlier (NULL for first N rows per partition) |
| `lead_N` | Value from N rows later |
| `diff_N` | `value - lag_N(value)` |
| `pct_change_N` | `(value - lag_N(value)) / lag_N(value)` |
| `first_value` | First value in the partition by `order_by` |
| `last_value` | Last value in the partition by `order_by` |

```python
Feature(
    "revenue__lag_1_offset",
    Options(context={"partition_by": ["customer_id"], "order_by": "month"}),
)
```

All offset types require `order_by`; partition ordering is what "previous row" means.

---

## Shared semantics

- **NULL in offsets**: Offsets reference *other* rows, so the current row being NULL does not force a NULL result. `lag_N` on a NULL row returns whatever sits N rows earlier (possibly non-NULL). `first_value` and `last_value` skip NULL source values and return the first or last non-NULL in the partition; the result is NULL only when every value in the partition is NULL. A NULL result from `lag_N`/`lead_N` happens when the referenced position falls outside the partition (first or last N rows).
- **NULL in ranks**: NULL values in the order column sort *last* in both ascending and descending directions; they still receive a rank. For `top_N`/`bottom_N` those NULL-ordered rows get `False` whenever N is smaller than the partition size.
- **Partition boundaries**: Operations never cross partitions. `lag_1` at the first row of a partition yields NULL, not a value from the previous partition.
- **Empty partitions**: Impossible by construction (a partition exists only if a row has its key).

---

## Related

- [Row-preserving contract](02-row-preserving-contract.md) - Analytic windows must preserve input order.
- [Window aggregation](06-window-aggregation.md) - Aggregates broadcast per partition; the same `partition_by` semantics apply.
- [Scalar and frame aggregate](08-scalar-and-frame-aggregate.md) - Aggregates over the whole frame, rolling, or expanding windows.
