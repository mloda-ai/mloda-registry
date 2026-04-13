# Overview

Data operations are feature groups that transform existing columns using declarative feature names. They ship in `mloda.community` and run unchanged across every supported compute framework.

**What**: Three categories of built-in feature groups: row-preserving analytics, row-reducing aggregations, and element-wise string transforms.
**When**: You want common analytic transforms (binning, windows, rolling aggregates, group-bys, string cleanup) without writing a custom feature group.
**Why**: A single feature name like `value_int__sum_agg` produces the same result on PyArrow, Pandas, Polars, DuckDB, and SQLite. Users pick the framework; the operation contract is identical.
**Where**: `mloda/community/feature_groups/data_operations/{row_preserving,aggregation,string}/`.
**How**: Request the operation as a feature (e.g. `value_int__p95_percentile`), pass any extra parameters via `Options(context=...)`, and call `mloda.run_all()`.

---

## The three categories

| Category | Location | Row behavior | Examples |
|---|---|---|---|
| Row-preserving | `row_preserving/` | Output row count and order match input | binning, window aggregation, rank, offset, percentile, scalar aggregate, frame aggregate, datetime |
| Aggregation | `aggregation/` | Reduces to one row per group | sum, avg, count, min, max, std, var, median, mode, nunique, first, last |
| String | `string/` | Row-preserving, element-wise on strings | upper, lower, trim, length, reverse |

Row-preserving is the largest category because analytic transforms that broadcast a computed value back onto each row (ranks, running totals, bin labels) all share the same invariant.

---

## Naming patterns

Every data operation encodes its parameters directly in the feature name. The prefix pattern is how the matcher locates the right feature group.

| Category | Pattern | Example |
|---|---|---|
| Aggregation | `{col}__{agg}_agg` | `value_int__sum_agg` |
| Window aggregation | `{col}__{agg}_window` | `value_int__avg_window` |
| Scalar aggregate | `{col}__{agg}_scalar` | `value_int__max_scalar` |
| Rank | `{col}__{rank_type}_ranked` | `score__dense_rank_ranked` |
| Offset | `{col}__{offset_type}_offset` | `value__lag_1_offset` |
| Percentile | `{col}__p{N}_percentile` | `latency__p95_percentile` |
| Frame aggregate | multiple forms | `value__sum_rolling_3`, `value__avg_7_day_window`, `value__cumsum`, `value__expanding_avg` |
| Binning | `{col}__{bin_op}_{N}` | `value_int__bin_5`, `value_int__qbin_10` |
| DateTime | `{col}__{part}` | `ts__year`, `ts__dayofweek` |
| String | `{col}__{str_op}` | `name__upper`, `text__length` |

The full pattern regex for each category lives in the corresponding `base.py`. For example:

```python
from mloda.community.feature_groups.data_operations.aggregation.base import AggregationFeatureGroup
from mloda.community.feature_groups.data_operations.string.base import StringFeatureGroup

AggregationFeatureGroup.PREFIX_PATTERN  # r".*__([\w]+)_agg$"
StringFeatureGroup.PREFIX_PATTERN       # r".+__(upper|lower|trim|length|reverse)$"
```

---

## A minimal example

```python
from mloda.user import Feature, Options, PluginLoader, mloda

PluginLoader.all()

features = [
    Feature("value_int__sum_agg", Options(context={"partition_by": ["region"]})),
    Feature("value_int__p95_percentile", Options(context={"partition_by": ["region"]})),
    Feature("name__upper"),
]

result = mloda.run_all(features, compute_frameworks={"PandasDataFrame"})
```

Each feature name resolves to one of the built-in data-operation feature groups. The `partition_by` option is consumed by the base class; no framework-specific code runs in user space.

---

## Where to go next

- [Row-preserving contract](02-row-preserving-contract.md) explains the central invariant that shapes most of this section.
- [Reference implementation pattern](03-reference-implementation.md) explains why PyArrow is the source of truth.
- [Known divergences](known-divergences.md) lists every audited case where a framework would otherwise produce a different result, and the mitigation for each.
- [Adding a new data operation](10-adding-new-operation.md) is the end-to-end recipe if you want to extend this system.
