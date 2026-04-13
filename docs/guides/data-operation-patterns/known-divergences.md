# Known Divergences

Authoritative list of cases where a non-PyArrow framework would, without intervention, produce a different result from the PyArrow reference on realistic inputs. For each one, this page records the mitigation and how to detect a regression.

**What**: Every audited case where a framework's native operator diverges from PyArrow semantics.
**When**: Read this before adding a new data-operation implementation, a new framework, or changing null or tie-breaking behavior in an existing one.
**Why**: Divergences of this kind are the most dangerous class of bug: the feature resolves, the pipeline succeeds, the output is silently wrong. Keeping a single list prevents that category from growing unnoticed.
**Where**: Audit of the 10 data operations under `mloda/community/feature_groups/data_operations/`.
**How**: Each entry records the divergence, the mitigation, and the test (or `supported_ops()` exclusion) that keeps it from silently regressing.

---

## Categories of divergence

Divergences fall into three kinds, handled in three different places.

| Kind | Example | Mitigation |
|---|---|---|
| Implementation fix | Polars `sum()` returns `0` for an all-null group; PyArrow returns `null`. | The framework implementation detects the edge case and returns the PyArrow-equivalent result. |
| Excluded op | SQLite `UPPER`/`LOWER` are ASCII-only; `REVERSE` has no native function. | `_validate_string_match` refuses to resolve the feature; `supported_ops()` skips the corresponding tests. See [Supported ops](04-supported-ops.md). |
| Accepted tolerance | Float accumulation order differs between columnar reductions and SQL window functions. | Cross-framework comparison uses `pytest.approx(rel=1e-6)` when the test flips `use_approx=True`. |
| Implementation guard | Several row-preserving implementations tag rows with hardcoded helper columns (`__mloda_rn__`, `__mloda_orig_idx`, ...). A user column with the same name would collide silently. | Every guarded `_compute_*` calls `assert_no_reserved_columns()`; any input column starting with the reserved prefix raises a clear `ValueError`. |

An entry is added here only after a cross-framework test or an explicit audit has confirmed the divergence.

---

## Entries

### Polars `sum()` on an all-null group returns `0`

- **Operations**: `aggregation`, `scalar_aggregate`, `window_aggregation`.
- **Where it lives**: `mloda/community/feature_groups/data_operations/aggregation/polars_lazy_aggregation.py`, `.../row_preserving/scalar_aggregate/polars_lazy_scalar_aggregate.py`, `.../row_preserving/window_aggregation/polars_lazy_window_aggregation.py`.
- **Reference behavior**: PyArrow's `pc.sum` returns `null` when every input value in the group is null.
- **Native Polars behavior**: `pl.col(...).sum()` returns `0` for the same input.
- **Mitigation kind**: Implementation fix.
- **How**: The Polars implementation wraps the `sum` expression with `pl.when(count > 0).then(sum).otherwise(None)`, so an all-null group maps back to `null`.
- **Regression signal**: The canonical 12-row fixture has a `score` column that is all-null. `test_null_policy_skip_all_null_column` in `mloda/testing/feature_groups/data_operations/aggregation/aggregation.py` asserts `score__sum_agg` is all-null per region, and fails if this correction is removed.

### Polars `rank()` returns null for null inputs

- **Operations**: `rank` (all rank types: `row_number`, `rank`, `dense_rank`, `percent_rank`, `ntile_N`, `top_N`, `bottom_N`).
- **Where it lives**: `mloda/community/feature_groups/data_operations/row_preserving/rank/polars_lazy_rank.py`.
- **Reference behavior**: PyArrow and every SQL engine assign null rows a real integer rank at the end of the ordering (nulls-last).
- **Native Polars behavior**: `pl.col(x).rank(...)` propagates nulls: null rows get `null` rank.
- **Mitigation kind**: Implementation fix.
- **How**: An internal `_NULL_FLAG_COL` helper counts nulls per partition; null rows are assigned `non_null_count + k` where `k` depends on the rank method. See `_row_number_nulls_last` and the `rank_type` branches in `polars_lazy_rank.py`.
- **Regression signal**: Group B in the canonical fixture has `value_int = [None, 50, 30, 60]`. Tests like `test_row_number_ranked`, `test_rank_ranked`, `test_dense_rank_ranked`, `test_percent_rank_ranked` in `mloda/testing/feature_groups/data_operations/row_preserving/rank/rank.py` assert the null row receives the last rank integer, not a null.

### Mode tie-breaking by first occurrence

- **Operations**: `aggregation` (`mode` agg type), `window_aggregation` (`mode` agg type).
- **Where it lives**: `mloda/community/feature_groups/data_operations/polars_mode_helpers.py` (shared Polars Lazy helpers used by both `polars_lazy_aggregation.py` and `polars_lazy_window_aggregation.py`); Pandas uses the vectorized `compute_mode_winners` helper in `pandas_helpers.py`.
- **Reference behavior**: PyArrow's `pc.mode` breaks ties by first occurrence in the input ordering.
- **Native framework behavior**: Polars' `.mode()` and Pandas' `.mode()` break ties differently (sorted order / multiple returned values / unspecified).
- **Mitigation kind**: Implementation fix.
- **How**: Both frameworks explicitly rank candidate values by `(count desc, first_occurrence_index asc)` and take the head. The Polars Lazy implementation stays inside the lazy / vectorised path: it adds per-`(partition, value)` count and first-index columns via `.over()`, then uses `sort_by([cnt, first_idx], descending=[True, False], maintain_order=True).first()` (no Python callback). On Pandas this is a single vectorized groupby over `(partition_by, value)` that aggregates count and first-occurrence index, avoiding a per-group Python reducer.
- **Regression signal**: The canonical fixture has values that tie; mode tests compare against the PyArrow reference via `_compare_with_reference`.

### SQLite `UPPER`/`LOWER` are ASCII-only; no native `REVERSE`

- **Operations**: `string` (`upper`, `lower`, `reverse`).
- **Where it lives**: `mloda/community/feature_groups/data_operations/string/sqlite_string.py`.
- **Reference behavior**: PyArrow's `pc.utf8_upper("héllo")` is `"HÉLLO"`.
- **Native SQLite behavior**: `UPPER('héllo')` returns `'HéLLO'`. `REVERSE` is not implemented.
- **Mitigation kind**: Excluded op.
- **How**: `SqliteStringOps._validate_string_match` returns `True` only for `trim` and `length`. Requesting `name__upper`, `name__lower`, or `name__reverse` with `compute_frameworks={"SqliteRelation"}` refuses to match at resolution time. The test class mirrors the decision through `supported_ops()`.
- **Regression signal**: `test_sqlite.py` inherits the unicode expected values (row 10 = `"héllo"` / `"HÉLLO"` / `"oll\u00e9h"`) and `supported_ops()` restricts the test suite to `{"trim", "length"}`. Adding an op without also enabling a Unicode-safe expression is caught immediately by cross-framework comparison.
- **Related**: Resolved from #146 via #147.

### Float accumulation order across SQL engines vs. columnar reductions

- **Operations**: `aggregation`, `scalar_aggregate`, `window_aggregation`, `percentile` (`avg`, `mean`, `std`, `var`, percentile interpolation).
- **Where it lives**: Integration tests that flip `use_approx=True` on the cross-framework comparison (e.g. `aggregation/tests/test_integration.py:96`, `scalar_aggregate/tests/test_integration.py:82`, `window_aggregation/tests/test_integration.py:81`, `percentile/tests/test_integration.py:78`).
- **Reference behavior**: PyArrow computes a columnar mean in a deterministic reduction tree.
- **Native SQL behavior**: DuckDB and SQLite accumulate with a running sum in query execution order, producing tiny relative-precision differences (~`1e-12` to `1e-8`).
- **Mitigation kind**: Accepted tolerance.
- **How**: The cross-framework assertion uses `pytest.approx(ref_value, rel=1e-6)` when the test's `use_approx` class attribute is `True`. Integer ops and null-equality still require exact match.
- **Regression signal**: If a change makes the relative error exceed `1e-6`, the approx check fails with a loud message pointing at the specific row.

### PyArrow lacks native `frame_aggregate`, `offset`, `percentile`, `rank`

- **Operations**: `row_preserving/frame_aggregate`, `.../offset`, `.../percentile`, `.../rank`.
- **Reference behavior**: PyArrow is the reference *for correctness semantics*, but it does not provide native rolling/expanding, LAG/LEAD, percentile, or rank. The reference implementations for these ops live in pure Python over PyArrow arrays.
- **Mitigation kind**: Excluded op (from the test suite, not from routing).
- **How**: The `supported_ops()` / `supported_agg_types()` on each operation's PyArrow test class returns an empty or reduced set so the suite does not try to compare against an implementation that does not exist. See `aggregation/tests/test_pyarrow.py` and `row_preserving/window_aggregation/tests/test_pyarrow.py`.
- **Regression signal**: Restoring the op on PyArrow requires both providing a native implementation and re-expanding the supported set; no silent skip is possible.
- **Related**: This is the "Category 1" case described in issue #146; listed here for completeness.

### Reserved `__mloda_` prefix for internal helper columns

- **Operations**: `aggregation` (Polars Lazy), `binning` (DuckDB, SQLite), `datetime` (SQLite), `frame_aggregate` (all frameworks), `offset` (all frameworks), `percentile` (Polars Lazy), `rank` (DuckDB, Polars Lazy, SQLite), `scalar_aggregate` (Polars Lazy), `string` (SQLite), `window_aggregation` (all frameworks).
- **Where it lives**: `mloda/community/feature_groups/data_operations/reserved_columns.py` defines `RESERVED_PREFIX = "__mloda_"` and the `assert_no_reserved_columns()` validator. Each guarded implementation calls it as the first statement of its `_compute_*` method.
- **Reference behavior**: PyArrow's reference implementation does not need helper columns and is silent about column names starting with `__mloda_`.
- **Native framework behavior**: Several row-preserving implementations add an internal helper column (for example `__mloda_rn__` to record original row order before a reordering window function). The helper name is hardcoded. If an input already carries a column with the same name, the helper would either overwrite user data or be dropped together with the helper at the end of the method, all without a diagnostic.
- **Mitigation kind**: Implementation guard.
- **How**: `assert_no_reserved_columns(<input columns>, framework=..., operation=...)` runs first in every guarded `_compute_*` method. It scans the input column names; any name starting with `__mloda_` raises `ValueError` with the offending name, the framework label, and the operation label. The whole prefix is reserved (not individual names) so future helpers added by new implementations are covered automatically.
- **Regression signal**: `mloda/community/feature_groups/data_operations/tests/test_reserved_columns.py` tests the helper. `mloda/community/feature_groups/data_operations/tests/test_reserved_columns_integration.py` exercises every guarded `_compute_*` with a colliding input and asserts the framework + operation label appear in the message; removing the guard from any implementation fails one test from that file.

### SQLite lacks `percentile` and `reverse` (and the string ops above)

- **Operations**: `row_preserving/percentile` and `string`.
- **Mitigation kind**: Excluded op.
- **How**: `supported_ops()` / `supported_agg_types()` on each SQLite test class restricts the covered set. See `row_preserving/percentile/tests/test_sqlite.py` and `string/tests/test_sqlite.py`.
- **Related**: Category 1 of issue #146.

---

## Audit coverage (2026-04-13)

The full audit covered all ten data operations: `binning`, `datetime`, `frame_aggregate`, `offset`, `percentile`, `rank`, `scalar_aggregate`, `window_aggregation`, `aggregation`, `string`. Every implementation file and every `*TestBase` was read.

| Operation | Frameworks audited | New divergence found? |
|---|---|---|
| aggregation | PyArrow, Pandas, Polars lazy, DuckDB, SQLite | No (all mitigated above) |
| binning | PyArrow, Pandas, Polars lazy, DuckDB, SQLite | No |
| datetime | PyArrow, Pandas, Polars lazy, DuckDB, SQLite | No |
| frame_aggregate | Pandas, Polars lazy, DuckDB, SQLite | No |
| offset | Pandas, Polars lazy, DuckDB, SQLite | No |
| percentile | Pandas, Polars lazy, DuckDB | No (float tolerance already accepted) |
| rank | Pandas, Polars lazy, DuckDB, SQLite | No (all mitigated above) |
| scalar_aggregate | PyArrow, Pandas, Polars lazy, DuckDB, SQLite | No |
| string | PyArrow, Pandas, Polars lazy, DuckDB, SQLite | No (SQLite ASCII mitigated) |
| window_aggregation | Pandas, Polars lazy, DuckDB, SQLite | No (all mitigated above) |

No unmitigated divergences were found. The `expected_*()` hooks defined on `StringTestBase` (`expected_upper`, `expected_lower`, `expected_trim`, `expected_length`, `expected_reverse`) are present for future use but are not currently overridden by any framework: after #147, SQLite no longer matches the unicode-unsafe ops instead of returning a divergent result.

---

## When to add to this page

Add a new entry here if and only if all three hold:

1. A framework operator produces a measurably different result from PyArrow on a realistic input.
2. That difference cannot be hidden by the cross-framework comparison (i.e. it would require an `expected_*()` override, a `use_approx=True` bump, a `pytest.skip`, or a `supported_ops()` exclusion).
3. The decision (fix vs. document vs. exclude) has been made and landed in code.

Do not add speculative entries. If an audit only uncovered a hypothetical divergence, add a failing regression test first so the entry corresponds to something the test suite measures.

---

## Related

- [Reference implementation pattern](03-reference-implementation.md) - Why PyArrow is authoritative.
- [Supported ops per framework](04-supported-ops.md) - The exclusion mechanism used when a framework cannot match PyArrow.
- [Row-preserving contract](02-row-preserving-contract.md) - The invariant every row-preserving op must honor.
