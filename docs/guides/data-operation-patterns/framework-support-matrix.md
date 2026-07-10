# Framework Support Matrix

One-page lookup for "does operation *X* work on framework *Y*?". Rows are the seventeen data operations (and their subtypes, where applicable); columns are the five compute frameworks mloda ships: PyArrow, Pandas, Polars lazy, DuckDB, SQLite.

**What**: A capability matrix rendered from the production capability declarations (`compute_framework_rule`, the `supports_compute_framework` hook, and match-time restrictions), queryable at runtime via `DataOperationsCatalog` (`from mloda.community.feature_groups.data_operations import DataOperationsCatalog`). If a cell is ✓, the framework's production implementation declares support for that subtype and the shared reference-based test suite runs it against that framework. If a cell is ✗, the implementation rejects the subtype at match time (see [Known divergences](known-divergences.md)).
**When**: Use before picking an op/framework pair, before adding a new framework implementation for an existing op, or while debugging why a feature resolves on one framework but skips on another.
**Why**: The authoritative information is the capability each concrete class declares in production code. `DataOperationsCatalog` flattens those declarations into one queryable structure; this page renders it as a single table and keeps it in sync via a drift check. The test-twin `supported_*()` sets are checked mirrors of the catalog, enforced by `tests/test_twin_catalog_consistency.py`.
**Where**: The tables below are guarded by a pytest drift check in `mloda/community/feature_groups/data_operations/tests/test_framework_support_matrix.py`. Do not edit the block between the `BEGIN GENERATED` and `END GENERATED` markers by hand. After changing any capability declaration, run `tox` (or at minimum that one test file). If the drift check fails, regenerate the block with a coding agent so its contents match what the test produces, then rerun until the test passes.
**How**: The test queries `DataOperationsCatalog`, renders the expected block, and asserts it matches the on-disk doc. CI runs `tox`, which runs the test, so a drifted matrix fails the build.

---

## Reading the tables

- The **summary** shows, per framework, whether an operation is fully supported (`full`), only partially supported (`partial (k/n)`), or absent (`--`).
- The **per-operation detail** tables show every subtype the operation defines, with ✓ / ✗ per cell. `--` means no production implementation ships for this framework (the catalog has no entry for it).
- A ✗ is not a bug. It is a deliberate exclusion declared by the production implementation at match time and documented in [Known divergences](known-divergences.md). The framework test class mirrors the exclusion with a `supported_*()` override in `*/tests/test_{framework}.py`, kept honest by `tests/test_twin_catalog_consistency.py`. See the matching divergence entry before attempting to add support.
- The matrix does not list every percentile quantile. Operations without a subtype axis (`percentile`, `ffill`, `ema`, `sessionization`, `resample`) render a single "(all)" row: the op either ships in full or does not ship at all. For `frame_aggregate` the per-frame-type detail breaks out time-window units (`time:second`, ..., `time:year`) so framework-specific unit gaps (e.g. SQLite/Pandas rejecting `time:month`) surface as ✗ rather than being hidden behind a single `time` row.
- For `frame_aggregate` the matrix axis is frame types only; aggregation-type support (`std`/`var`/`median`) is enforced by the `supports_compute_framework` hook rather than represented as a catalog subtype, because Polars' support is two-dimensional (it depends on the frame type: cumulative/expanding exclude `std`/`var`/`median` while rolling/time include them). See [Known divergences](known-divergences.md). The catalog deliberately stays single-axis: the shared `SubtypeCapabilityHook` (`supported_subtypes(secondary)`) is the authority for higher-dimensional (frame_type x agg_type) capability, and the catalog is not extended to multi-axis subtypes unless 2D operations proliferate.

## Querying capabilities at runtime

Everything in this page is available programmatically through `DataOperationsCatalog`, so you can check an op/framework pair before building features instead of reading tables:

```python
from mloda.community.feature_groups.data_operations import DataOperationsCatalog

# Exact cell: is median aggregation available on SQLite?
DataOperationsCatalog.is_supported("aggregation", "median", "SqliteFramework")  # False
DataOperationsCatalog.is_supported("aggregation", "mean", "SqliteFramework")    # True

# Operation-level: does percentile ship on DuckDB at all?
DataOperationsCatalog.is_supported("percentile", framework="DuckDBFramework")   # True

# Full record: naming pattern, subtype universe, per-framework support.
info = DataOperationsCatalog.get("aggregation")
info.prefix_pattern            # r".*__([\w]+)_agg$"
info.subtypes                  # ("sum", "avg", "mean", "count", ...)
info.frameworks["SqliteFramework"]  # frozenset({"sum", "avg", "mean", "count", "min", "max"})

for op in DataOperationsCatalog.list():
    print(op.name, op.prefix_pattern)
```

Framework names are the compute framework class names (`PyArrowTable`, `PandasDataFrame`, `PolarsLazyDataFrame`, `DuckDBFramework`, `SqliteFramework`), matched case-insensitively. Unknown operations and out-of-universe subtypes raise a `ValueError` listing the valid values; an unknown framework simply returns `False`.

To check a concrete feature name instead of an op/subtype pair, use core's `resolve_feature`; the same capability hook feeds both:

```python
from mloda.user import PluginLoader
from mloda.steward import resolve_feature

PluginLoader.all()
resolved = resolve_feature("value__median_scalar")
resolved.supported_compute_frameworks    # ["DuckDBFramework", "PandasDataFrame", ...]
resolved.unsupported_compute_frameworks  # ["SqliteFramework"]
```

Unsupported combinations are also rejected at match time when running a pipeline, so a request pinned to an incapable framework fails at planning with a message naming the supported frameworks, not at compute.

## Not in this matrix: joins

This matrix only covers data-operation feature groups (single-source column transforms). Joins are a separate concern handled by the compute frameworks' merge engines in core, not by registry feature groups, so they do not appear above. The **as-of (point-in-time) join** added in mloda 0.8.0 has its own per-backend support table and a worked example in [Links and joins](../feature-group-patterns/08-links-joins.md#as-of-point-in-time-joins).

---

<!-- BEGIN GENERATED: framework-support-matrix -->

## Summary

Cells reflect the production capability declarations (`compute_framework_rule`, the `supports_compute_framework` hook, and match-time restrictions), queryable via `DataOperationsCatalog`. `full` means the framework's production implementation declares support for every subtype this operation defines. `partial (k/n)` means it declares k of the n subtypes and rejects the rest. `--` means no implementation ships for this framework.

| Operation | PyArrow | Pandas | Polars lazy | DuckDB | SQLite |
|---|---|---|---|---|---|
| aggregation | partial (15/17) | full | full | full | partial (6/17) |
| binning | full | full | full | full | full |
| datetime | full | full | full | full | full |
| frame_aggregate | -- | partial (8/10) | full | full | partial (8/10) |
| offset | -- | full | full | full | full |
| percentile | -- | full | full | full | -- |
| rank | -- | full | full | full | full |
| scalar_aggregate | full | full | full | full | partial (6/13) |
| scalar_arithmetic | full | full | full | full | full |
| point_arithmetic | full | full | full | full | full |
| time_bucketization | full | full | full | full | full |
| ffill | full | full | full | full | full |
| ema | -- | full | full | -- | -- |
| sessionization | full | full | full | full | full |
| window_aggregation | partial (15/17) | full | full | full | partial (6/17) |
| string | full | full | full | full | partial (2/5) |
| resample | full | full | full | full | -- |

## Per-operation detail

✓ = the framework's production implementation declares support for this subtype. ✗ = the implementation rejects it. `--` = no implementation ships for this framework.

### aggregation

| Agg type | PyArrow | Pandas | Polars lazy | DuckDB | SQLite |
|---|---|---|---|---|---|
| `sum` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `avg` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `mean` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `count` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `min` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `max` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `std` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `var` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `std_pop` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `std_samp` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `var_pop` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `var_samp` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `median` | ✗ | ✓ | ✓ | ✓ | ✗ |
| `mode` | ✗ | ✓ | ✓ | ✓ | ✗ |
| `nunique` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `first` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `last` | ✓ | ✓ | ✓ | ✓ | ✗ |

### binning

| Op | PyArrow | Pandas | Polars lazy | DuckDB | SQLite |
|---|---|---|---|---|---|
| `bin` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `qbin` | ✓ | ✓ | ✓ | ✓ | ✓ |

### datetime

| Op | PyArrow | Pandas | Polars lazy | DuckDB | SQLite |
|---|---|---|---|---|---|
| `year` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `month` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `day` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `hour` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `minute` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `second` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `dayofweek` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `is_weekend` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `quarter` | ✓ | ✓ | ✓ | ✓ | ✓ |

### frame_aggregate

| Frame type | PyArrow | Pandas | Polars lazy | DuckDB | SQLite |
|---|---|---|---|---|---|
| `rolling` | -- | ✓ | ✓ | ✓ | ✓ |
| `time:second` | -- | ✓ | ✓ | ✓ | ✓ |
| `time:minute` | -- | ✓ | ✓ | ✓ | ✓ |
| `time:hour` | -- | ✓ | ✓ | ✓ | ✓ |
| `time:day` | -- | ✓ | ✓ | ✓ | ✓ |
| `time:week` | -- | ✓ | ✓ | ✓ | ✓ |
| `time:month` | -- | ✗ | ✓ | ✓ | ✗ |
| `time:year` | -- | ✗ | ✓ | ✓ | ✗ |
| `cumulative` | -- | ✓ | ✓ | ✓ | ✓ |
| `expanding` | -- | ✓ | ✓ | ✓ | ✓ |

### offset

| Offset type | PyArrow | Pandas | Polars lazy | DuckDB | SQLite |
|---|---|---|---|---|---|
| `lag` | -- | ✓ | ✓ | ✓ | ✓ |
| `lead` | -- | ✓ | ✓ | ✓ | ✓ |
| `diff` | -- | ✓ | ✓ | ✓ | ✓ |
| `pct_change` | -- | ✓ | ✓ | ✓ | ✓ |
| `first_value` | -- | ✓ | ✓ | ✓ | ✓ |
| `last_value` | -- | ✓ | ✓ | ✓ | ✓ |

### percentile

| Op | PyArrow | Pandas | Polars lazy | DuckDB | SQLite |
|---|---|---|---|---|---|
| (all) | -- | ✓ | ✓ | ✓ | -- |

### rank

| Rank type | PyArrow | Pandas | Polars lazy | DuckDB | SQLite |
|---|---|---|---|---|---|
| `row_number` | -- | ✓ | ✓ | ✓ | ✓ |
| `rank` | -- | ✓ | ✓ | ✓ | ✓ |
| `dense_rank` | -- | ✓ | ✓ | ✓ | ✓ |
| `percent_rank` | -- | ✓ | ✓ | ✓ | ✓ |
| `ntile` | -- | ✓ | ✓ | ✓ | ✓ |
| `top` | -- | ✓ | ✓ | ✓ | ✓ |
| `bottom` | -- | ✓ | ✓ | ✓ | ✓ |

### scalar_aggregate

| Agg type | PyArrow | Pandas | Polars lazy | DuckDB | SQLite |
|---|---|---|---|---|---|
| `sum` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `min` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `max` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `avg` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `mean` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `count` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `std` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `var` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `std_pop` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `std_samp` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `var_pop` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `var_samp` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `median` | ✓ | ✓ | ✓ | ✓ | ✗ |

### scalar_arithmetic

| Op | PyArrow | Pandas | Polars lazy | DuckDB | SQLite |
|---|---|---|---|---|---|
| `add` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `subtract` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `multiply` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `divide` | ✓ | ✓ | ✓ | ✓ | ✓ |

### point_arithmetic

| Op | PyArrow | Pandas | Polars lazy | DuckDB | SQLite |
|---|---|---|---|---|---|
| `add` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `subtract` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `multiply` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `divide` | ✓ | ✓ | ✓ | ✓ | ✓ |

### time_bucketization

| Op | PyArrow | Pandas | Polars lazy | DuckDB | SQLite |
|---|---|---|---|---|---|
| `floor` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `ceil` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `round` | ✓ | ✓ | ✓ | ✓ | ✓ |

### ffill

| Op | PyArrow | Pandas | Polars lazy | DuckDB | SQLite |
|---|---|---|---|---|---|
| (all) | ✓ | ✓ | ✓ | ✓ | ✓ |

### ema

| Op | PyArrow | Pandas | Polars lazy | DuckDB | SQLite |
|---|---|---|---|---|---|
| (all) | -- | ✓ | ✓ | -- | -- |

### sessionization

| Op | PyArrow | Pandas | Polars lazy | DuckDB | SQLite |
|---|---|---|---|---|---|
| (all) | ✓ | ✓ | ✓ | ✓ | ✓ |

### window_aggregation

| Agg type | PyArrow | Pandas | Polars lazy | DuckDB | SQLite |
|---|---|---|---|---|---|
| `sum` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `avg` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `mean` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `count` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `min` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `max` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `std` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `var` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `std_pop` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `std_samp` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `var_pop` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `var_samp` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `median` | ✗ | ✓ | ✓ | ✓ | ✗ |
| `mode` | ✗ | ✓ | ✓ | ✓ | ✗ |
| `nunique` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `first` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `last` | ✓ | ✓ | ✓ | ✓ | ✗ |

### string

| Op | PyArrow | Pandas | Polars lazy | DuckDB | SQLite |
|---|---|---|---|---|---|
| `upper` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `lower` | ✓ | ✓ | ✓ | ✓ | ✗ |
| `trim` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `length` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `reverse` | ✓ | ✓ | ✓ | ✓ | ✗ |

### resample

| Op | PyArrow | Pandas | Polars lazy | DuckDB | SQLite |
|---|---|---|---|---|---|
| (all) | ✓ | ✓ | ✓ | ✓ | -- |

<!-- END GENERATED: framework-support-matrix -->

---

## When a cell is ✗ or `--`

The three divergence kinds in [Known divergences](known-divergences.md) map to the detail tables as follows:

- **Excluded subtype → ✗** (SQLite `upper` / `lower` / `reverse`, SQLite `percentile`, PyArrow aggregation/window `median` / `mode`): the framework has a test class for this operation, but the implementation refuses to match at resolution time and the test class's `supported_*()` override mirrors the refusal so inherited tests skip cleanly.
- **Missing framework → `--`** (PyArrow `frame_aggregate`, `offset`, `percentile`, `rank`): no production implementation exists, and the framework has no test class for the operation. The operation requires native rolling / LAG / percentile / rank that PyArrow does not provide, and the reference implementation lives in pure Python over PyArrow arrays. Adding it requires a real framework implementation, not just relaxing an exclusion.
- **Tolerance constrained, not marked**: float-accumulation tolerance is not exposed as ✗ or `--`; it shows up as `use_approx=True` on the relevant cross-framework assertions. See the "Float accumulation order" entry in Known divergences.

To promote a ✗ to ✓:

1. Read the Known-divergences entry for the cell to understand why it is ✗.
2. Implement the missing native expression (or remove the implementation-side `_validate_*_match` exclusion).
3. Update the framework test class's `supported_*()` override, or remove it if the framework now matches the full default set.
4. Rerun the pytest drift check (`pytest mloda/community/feature_groups/data_operations/tests/test_framework_support_matrix.py`). If it fails, regenerate the `BEGIN GENERATED` / `END GENERATED` block with a coding agent and rerun.
5. Run `PYTEST_WORKERS=1 tox` to confirm the inherited tests now pass.

---

## Related

- [Supported ops per framework](04-supported-ops.md) - the mechanism (`supported_ops()`, `_validate_*_match`) that backs every cell in this matrix.
- [Known divergences](known-divergences.md) - the divergence-by-divergence rationale behind every ✗ in the detail tables.
- [Reference implementation pattern](03-reference-implementation.md) - why PyArrow is the reference for each ✓ in the matrix.
- [Adding a new data operation](10-adding-new-operation.md) - how to add a new row to this matrix from scratch.
