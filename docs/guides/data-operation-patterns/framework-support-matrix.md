# Framework Support Matrix

One-page lookup for "does operation *X* work on framework *Y*?". Rows are the ten data operations (and their subtypes, where applicable); columns are the five compute frameworks mloda ships: PyArrow, Pandas, Polars lazy, DuckDB, SQLite.

**What**: A capability matrix that mirrors the `supported_*()` class methods declared on every framework test class. If a cell is ✓, the shared reference-based test suite runs that subtype against that framework and compares the result to PyArrow. If a cell is ✗, the framework either cannot express that subtype or has been deliberately excluded (see [Known divergences](known-divergences.md)).
**When**: Use before picking an op/framework pair, before adding a new framework implementation for an existing op, or while debugging why a feature resolves on one framework but skips on another.
**Why**: The authoritative information lives in `supported_agg_types()` / `supported_ops()` / `supported_offset_types()` / `supported_rank_types()` overrides scattered across ten operation directories. This page flattens them into a single table and keeps it in sync via a drift check.
**Where**: The tables below are guarded by a pytest drift check in `mloda/community/feature_groups/data_operations/tests/test_framework_support_matrix.py`. Do not edit the block between the `BEGIN GENERATED` and `END GENERATED` markers by hand. After changing any `supported_*()` override, run `tox` (or at minimum that one test file). If the drift check fails, regenerate the block with a coding agent so its contents match what the test produces, then rerun until the test passes.
**How**: The test reflects each framework's test class, reads its `supported_*()` set, renders the expected block, and asserts it matches the on-disk doc. CI runs `tox`, which runs the test, so a drifted matrix fails the build.

---

## Reading the tables

- The **summary** shows, per framework, whether an operation is fully covered (`full`), only partially covered (`partial (k/n)`), or absent (`--`).
- The **per-operation detail** tables show every subtype that any framework supports, with ✓ / ✗ per cell. `--` means that framework has no test class (and therefore no implementation) for this operation.
- A ✗ is not a bug. It is a deliberate exclusion documented in [Known divergences](known-divergences.md) or recorded by a `supported_*()` override in `*/tests/test_{framework}.py`. See the matching entry there before attempting to add support.
- The matrix does not list every percentile quantile or frame-aggregate window kind. For those the op either ships in full or does not ship at all: see the detail tables for `percentile`, `datetime`, and `frame_aggregate`, where the single "(all)" row reflects that absence of a `supported_*()` method.

---

<!-- BEGIN GENERATED: framework-support-matrix -->

## Summary

`full` means every subtype listed in the per-operation table below is supported. `partial (k/n)` means the framework's test class restricts `supported_*()` to k of the n subtypes this operation defines. `--` means the framework has no test class for this operation (typically because no production implementation exists).

| Operation | PyArrow | Pandas | Polars lazy | DuckDB | SQLite |
|---|---|---|---|---|---|
| aggregation | partial (15/17) | partial (16/17) | partial (16/17) | partial (16/17) | partial (5/17) |
| binning | full | full | full | full | full |
| datetime | full | full | full | full | full |
| frame_aggregate | -- | full | full | full | full |
| offset | -- | full | full | full | partial (4/6) |
| percentile | -- | full | full | full | -- |
| rank | -- | full | full | full | full |
| scalar_aggregate | full | full | full | full | partial (6/13) |
| window_aggregation | partial (15/17) | partial (16/17) | partial (16/17) | partial (16/17) | partial (5/17) |
| string | full | full | full | full | partial (2/5) |

## Per-operation detail

✓ = the framework's test-class `supported_*()` includes this subtype. ✗ = excluded. `--` = no test class for this framework.

### aggregation

| Agg type | PyArrow | Pandas | Polars lazy | DuckDB | SQLite |
|---|---|---|---|---|---|
| `sum` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `avg` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `mean` | ✓ | ✗ | ✗ | ✗ | ✗ |
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
| (all) | ✓ | ✓ | ✓ | ✓ | ✓ |

### frame_aggregate

| Frame type | PyArrow | Pandas | Polars lazy | DuckDB | SQLite |
|---|---|---|---|---|---|
| (all) | -- | ✓ | ✓ | ✓ | ✓ |

### offset

| Offset type | PyArrow | Pandas | Polars lazy | DuckDB | SQLite |
|---|---|---|---|---|---|
| `lag` | -- | ✓ | ✓ | ✓ | ✓ |
| `lead` | -- | ✓ | ✓ | ✓ | ✓ |
| `diff` | -- | ✓ | ✓ | ✓ | ✗ |
| `pct_change` | -- | ✓ | ✓ | ✓ | ✗ |
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

### scalar_aggregate

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
| `median` | ✓ | ✓ | ✓ | ✓ | ✗ |

### window_aggregation

| Agg type | PyArrow | Pandas | Polars lazy | DuckDB | SQLite |
|---|---|---|---|---|---|
| `sum` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `avg` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `mean` | ✓ | ✗ | ✗ | ✗ | ✗ |
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

<!-- END GENERATED: framework-support-matrix -->

---

## When a cell is ✗ or `--`

The three divergence kinds in [Known divergences](known-divergences.md) map to the detail tables as follows:

- **Excluded subtype → ✗** (SQLite `upper` / `lower` / `reverse`, SQLite `percentile`, SQLite offset `diff` / `pct_change`, PyArrow aggregation/window `median` / `mode`): the framework has a test class for this operation, but the implementation refuses to match at resolution time and the test class's `supported_*()` override mirrors the refusal so inherited tests skip cleanly.
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
