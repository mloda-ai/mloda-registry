# Known Divergences

Authoritative list of cases where a non-PyArrow framework would, without intervention, produce a different result from the PyArrow reference on realistic inputs. For each one, this page records the mitigation and how to detect a regression.

**What**: Every audited case where a framework's native operator diverges from PyArrow semantics.
**When**: Read this before adding a new data-operation implementation, a new framework, or changing null or tie-breaking behavior in an existing one.
**Why**: Divergences of this kind are the most dangerous class of bug: the feature resolves, the pipeline succeeds, the output is silently wrong. Keeping a single list prevents that category from growing unnoticed.
**Where**: Audit of the 11 data operations under `mloda/community/feature_groups/data_operations/`.
**How**: Each entry records the divergence, the mitigation, and the test (or `supported_ops()` exclusion) that keeps it from silently regressing. Every entry also carries a machine-checked block (see [When to add to this page](#when-to-add-to-this-page)) so a `tox` test can verify the cited regression test still exists.

---

## Categories of divergence

Divergences fall into three kinds, handled in three different places.

| Kind | Example | Mitigation |
|---|---|---|
| Implementation fix | Polars `sum()` returns `0` for an all-null group; PyArrow returns `null`. | The framework implementation detects the edge case and returns the PyArrow-equivalent result. |
| Excluded op | SQLite `UPPER`/`LOWER` are ASCII-only; `REVERSE` has no native function. | `_validate_string_match` refuses to resolve the feature; `supported_ops()` skips the corresponding tests. See [Supported ops](04-supported-ops.md). |
| Accepted tolerance | Float accumulation order differs between columnar reductions and SQL window functions. | Cross-framework comparison uses `pytest.approx(rel=1e-6)` when the test flips `use_approx=True`. |
| Collision-free helper naming | Row-preserving implementations tag rows with helper columns (to restore input order after a reordering window function). A user column with the same name would collide silently. | Every backend picks a collision-free helper name at runtime, so any user column name is accepted. The SQL backends (DuckDB, SQLite) use `pick_helper_column_name`; the pandas/polars/pyarrow backends use `unique_helper_name` (`helper_columns.py`). The old `__mloda_` reject-guard was removed in #221. |

An entry is added here only after a cross-framework test or an explicit audit has confirmed the divergence.

---

## Entries

### Polars `sum()` on an all-null group returns `0`

<!-- machine-checked
operation: aggregation, scalar_aggregate, window_aggregation
framework: polars_lazy
condition: sum() on an all-null group returns 0 instead of PyArrow's null
mitigation_location:
- mloda/community/feature_groups/data_operations/aggregation/polars_lazy_aggregation.py
- mloda/community/feature_groups/data_operations/row_preserving/scalar_aggregate/polars_lazy_scalar_aggregate.py
- mloda/community/feature_groups/data_operations/row_preserving/window_aggregation/polars_lazy_window_aggregation.py
regression_test:
- mloda/testing/feature_groups/data_operations/aggregation/aggregation.py::AggregationTestBase::test_all_null_column_per_group
-->

- **Operations**: `aggregation`, `scalar_aggregate`, `window_aggregation`.
- **Where it lives**: `mloda/community/feature_groups/data_operations/aggregation/polars_lazy_aggregation.py`, `.../row_preserving/scalar_aggregate/polars_lazy_scalar_aggregate.py`, `.../row_preserving/window_aggregation/polars_lazy_window_aggregation.py`.
- **Reference behavior**: PyArrow's `pc.sum` returns `null` when every input value in the group is null.
- **Native Polars behavior**: `pl.col(...).sum()` returns `0` for the same input.
- **Mitigation kind**: Implementation fix.
- **How**: The Polars implementation wraps the `sum` expression with `pl.when(count > 0).then(sum).otherwise(None)`, so an all-null group maps back to `null`.
- **Regression signal**: The canonical 12-row fixture has a `score` column that is all-null. `test_all_null_column_per_group[sum]` in `mloda/testing/feature_groups/data_operations/aggregation/aggregation.py` asserts `score__sum_agg` is all-null per region, and fails if this correction is removed.

### Polars `rank()` returns null for null inputs

<!-- machine-checked
operation: rank
framework: polars_lazy
condition: rank() propagates null instead of assigning a nulls-last integer rank
mitigation_location:
- mloda/community/feature_groups/data_operations/row_preserving/rank/polars_lazy_rank.py
regression_test:
- mloda/testing/feature_groups/data_operations/row_preserving/rank/rank.py::RankTestBase::test_row_number_ranked
- mloda/testing/feature_groups/data_operations/row_preserving/rank/rank.py::RankTestBase::test_rank_ranked
- mloda/testing/feature_groups/data_operations/row_preserving/rank/rank.py::RankTestBase::test_dense_rank_ranked
- mloda/testing/feature_groups/data_operations/row_preserving/rank/rank.py::RankTestBase::test_percent_rank_ranked
-->

- **Operations**: `rank` (all rank types: `row_number`, `rank`, `dense_rank`, `percent_rank`, `ntile_N`, `top_N`, `bottom_N`).
- **Where it lives**: `mloda/community/feature_groups/data_operations/row_preserving/rank/polars_lazy_rank.py`.
- **Reference behavior**: PyArrow and every SQL engine assign null rows a real integer rank at the end of the ordering (nulls-last).
- **Native Polars behavior**: `pl.col(x).rank(...)` propagates nulls: null rows get `null` rank.
- **Mitigation kind**: Implementation fix.
- **How**: An internal `_NULL_FLAG_COL` helper counts nulls per partition; null rows are assigned `non_null_count + k` where `k` depends on the rank method. See `_row_number_nulls_last` and the `rank_type` branches in `polars_lazy_rank.py`.
- **Regression signal**: Group B in the canonical fixture has `value_int = [None, 50, 30, 60]`. Tests like `test_row_number_ranked`, `test_rank_ranked`, `test_dense_rank_ranked`, `test_percent_rank_ranked` in `mloda/testing/feature_groups/data_operations/row_preserving/rank/rank.py` assert the null row receives the last rank integer, not a null.

### Mode tie-breaking by first occurrence

<!-- machine-checked
operation: aggregation, window_aggregation
framework: polars_lazy, pandas
condition: native mode() breaks ties differently from PyArrow's first-occurrence rule
mitigation_location:
- mloda/community/feature_groups/data_operations/polars_mode_helpers.py
- mloda/community/feature_groups/data_operations/pandas_helpers.py
regression_test:
- mloda/testing/feature_groups/data_operations/aggregation/aggregation.py::AggregationTestBase::test_cross_framework_agg
- mloda/testing/feature_groups/data_operations/row_preserving/window_aggregation/window_aggregation.py::WindowAggregationTestBase::test_cross_framework
-->

- **Operations**: `aggregation` (`mode` agg type), `window_aggregation` (`mode` agg type).
- **Where it lives**: `mloda/community/feature_groups/data_operations/polars_mode_helpers.py` (shared Polars Lazy helpers used by both `polars_lazy_aggregation.py` and `polars_lazy_window_aggregation.py`); Pandas uses the vectorized `compute_mode_winners` helper in `pandas_helpers.py`.
- **Reference behavior**: PyArrow's `pc.mode` breaks ties by first occurrence in the input ordering.
- **Native framework behavior**: Polars' `.mode()` and Pandas' `.mode()` break ties differently (sorted order / multiple returned values / unspecified).
- **Mitigation kind**: Implementation fix.
- **How**: Both frameworks explicitly rank candidate values by `(count desc, first_occurrence_index asc)` and take the head. The Polars Lazy implementation stays inside the lazy / vectorised path: it adds per-`(partition, value)` count and first-index columns via `.over()`, then uses `sort_by([cnt, first_idx], descending=[True, False], maintain_order=True).first()` (no Python callback). On Pandas this is a single vectorized groupby over `(partition_by, value)` that aggregates count and first-occurrence index, avoiding a per-group Python reducer.
- **Regression signal**: The canonical fixture has values that tie; `test_cross_framework_agg[mode]` and `test_cross_framework[mode]` compare against the PyArrow reference via `_compare_with_reference`.

### SQLite divide-by-zero returns NULL instead of IEEE-754 inf/nan

<!-- machine-checked
operation: point_arithmetic
framework: sqlite
condition: divide-by-zero on float operands returns NULL instead of inf/-inf/nan
mitigation_location:
- mloda/community/feature_groups/data_operations/row_preserving/point_arithmetic/sqlite_point_arithmetic.py
regression_test:
- mloda/community/feature_groups/data_operations/row_preserving/point_arithmetic/tests/test_security.py::TestDivideByZeroPerRow::test_sqlite_divide_by_zero_per_row
-->

- **Operations**: `point_arithmetic` (`divide` op).
- **Where it lives**: `mloda/community/feature_groups/data_operations/row_preserving/point_arithmetic/sqlite_point_arithmetic.py`.
- **Reference behavior**: PyArrow's `pc.divide` on float64 operands returns IEEE-754 `inf` / `-inf` for `N/0` (sign of `N`) and `NaN` for `0/0`. Pandas, Polars lazy, and DuckDB all match this when both operands are cast to float / DOUBLE.
- **Native SQLite behavior**: `CAST(a AS REAL) / CAST(b AS REAL)` returns `NULL` for any divide-by-zero or null operand. SQLite has no native IEEE-754 inf/nan storage; the engine substitutes `NULL` for results that would otherwise be a non-finite float.
- **Mitigation kind**: Accepted divergence (no mitigation attempted; the contract is documented and the test base accommodates both behaviors).
- **How**: `PointArithmeticTestBase.divide_zero_propagates_inf()` is `True` by default. `TestSqlitePointArithmetic` overrides it to `False`, so the cross-framework divide-by-zero row at index 5 of the canonical fixture (`value_int=50, amount=0.0`) is asserted as `inf` on the four other backends and `None` on SQLite. Forcing SQLite into inf-emitting behavior would require an out-of-band float library; the cost outweighs the benefit for an operation whose primary callers will pick a non-SQLite backend when they need IEEE-754 semantics anyway.
- **Regression signal**: `TestDivideByZeroPerRow.test_sqlite_divide_by_zero_per_row` in `mloda/community/feature_groups/data_operations/row_preserving/point_arithmetic/tests/test_security.py` pins the SQLite-specific `NULL` expectation; the corresponding four backends pin `inf`/`-inf`/`nan` per the truth table. If a future SQLite implementation begins returning a different non-NULL value, the assertion fails.
- **Scope note**: `scalar_arithmetic` does not appear here because its divisor is a validated `Options` constant; `divide_by_zero` is rejected up front before dispatch reaches any backend, so the per-row divergence cannot arise there.

### SQLite `UPPER`/`LOWER` are ASCII-only; no native `REVERSE`

<!-- machine-checked
operation: string
framework: sqlite
condition: UPPER/LOWER are ASCII-only and REVERSE has no native function
mitigation_location:
- mloda/community/feature_groups/data_operations/string/sqlite_string.py
regression_test:
- mloda/community/feature_groups/data_operations/string/tests/test_sqlite.py::TestSqliteUnsupportedOps::test_unsupported_op_does_not_match
-->

- **Operations**: `string` (`upper`, `lower`, `reverse`).
- **Where it lives**: `mloda/community/feature_groups/data_operations/string/sqlite_string.py`.
- **Reference behavior**: PyArrow's `pc.utf8_upper("héllo")` is `"HÉLLO"`.
- **Native SQLite behavior**: `UPPER('héllo')` returns `'HéLLO'`. `REVERSE` is not implemented.
- **Mitigation kind**: Excluded op.
- **How**: `SqliteStringOps._validate_string_match` returns `True` only for `trim` and `length`. Requesting `name__upper`, `name__lower`, or `name__reverse` with `compute_frameworks={"SqliteRelation"}` refuses to match at resolution time. The test class mirrors the decision through `supported_ops()`.
- **Regression signal**: `test_unsupported_op_does_not_match[upper|lower|reverse]` pins the refusal, and `test_sqlite.py` inherits the unicode expected values (row 10 = `"héllo"` / `"HÉLLO"` / `"oll\u00e9h"`) and `supported_ops()` restricts the test suite to `{"trim", "length"}`. Adding an op without also enabling a Unicode-safe expression is caught immediately by cross-framework comparison.
- **Related**: Resolved from #146 via #147.

### Float accumulation order across SQL engines vs. columnar reductions

<!-- machine-checked
operation: aggregation, scalar_aggregate, window_aggregation, percentile
framework: duckdb, sqlite
condition: SQL running-sum accumulation differs from PyArrow's reduction tree (~1e-12 to 1e-8)
mitigation_location:
- mloda/community/feature_groups/data_operations/aggregation/tests/test_integration.py
- mloda/community/feature_groups/data_operations/row_preserving/scalar_aggregate/tests/test_integration.py
- mloda/community/feature_groups/data_operations/row_preserving/window_aggregation/tests/test_integration.py
- mloda/community/feature_groups/data_operations/row_preserving/percentile/tests/test_integration.py
regression_test:
- mloda/community/feature_groups/data_operations/aggregation/tests/test_integration.py::TestAggregationIntegration
- mloda/community/feature_groups/data_operations/row_preserving/scalar_aggregate/tests/test_integration.py::TestScalarAggregateIntegration
- mloda/community/feature_groups/data_operations/row_preserving/window_aggregation/tests/test_integration.py::TestWindowAggregationIntegration
- mloda/community/feature_groups/data_operations/row_preserving/percentile/tests/test_integration.py::TestPercentileIntegration
-->

- **Operations**: `aggregation`, `scalar_aggregate`, `window_aggregation`, `percentile` (`avg`, `mean`, `std`, `var`, percentile interpolation).
- **Where it lives**: Integration tests that flip `use_approx=True` on the cross-framework comparison (e.g. `aggregation/tests/test_integration.py:96`, `scalar_aggregate/tests/test_integration.py:82`, `window_aggregation/tests/test_integration.py:81`, `percentile/tests/test_integration.py:78`).
- **Reference behavior**: PyArrow computes a columnar mean in a deterministic reduction tree.
- **Native SQL behavior**: DuckDB and SQLite accumulate with a running sum in query execution order, producing tiny relative-precision differences (~`1e-12` to `1e-8`).
- **Mitigation kind**: Accepted tolerance.
- **How**: The cross-framework assertion uses `pytest.approx(ref_value, rel=1e-6)` when the test's `use_approx` class attribute is `True`. Integer ops and null-equality still require exact match.
- **Regression signal**: If a change makes the relative error exceed `1e-6`, the approx check fails with a loud message pointing at the specific row.

### PyArrow lacks native `frame_aggregate`, `offset`, `percentile`, `rank`

<!-- machine-checked
operation: frame_aggregate, offset, percentile, rank
framework: pyarrow
condition: PyArrow has no native rolling/expanding, LAG/LEAD, percentile, or rank operator
mitigation_location:
- mloda/community/feature_groups/data_operations/aggregation/tests/test_pyarrow.py
- mloda/community/feature_groups/data_operations/row_preserving/window_aggregation/tests/test_pyarrow.py
regression_test:
- mloda/community/feature_groups/data_operations/aggregation/tests/test_pyarrow.py::TestPyArrowAggregation
- mloda/community/feature_groups/data_operations/row_preserving/window_aggregation/tests/test_pyarrow.py::TestPyArrowWindowAggregation
- mloda/community/feature_groups/data_operations/tests/test_framework_support_matrix.py::test_framework_support_matrix_is_in_sync
-->

- **Operations**: `row_preserving/frame_aggregate`, `.../offset`, `.../percentile`, `.../rank`.
- **Reference behavior**: PyArrow is the reference *for correctness semantics*, but it does not provide native rolling/expanding, LAG/LEAD, percentile, or rank. The reference implementations for these ops live in pure Python over PyArrow arrays.
- **Mitigation kind**: Excluded op (from the test suite, not from routing).
- **How**: The `supported_ops()` / `supported_agg_types()` on each operation's PyArrow test class returns an empty or reduced set so the suite does not try to compare against an implementation that does not exist. See `aggregation/tests/test_pyarrow.py` and `row_preserving/window_aggregation/tests/test_pyarrow.py`.
- **Regression signal**: Restoring the op on PyArrow requires both providing a native implementation and re-expanding the supported set; no silent skip is possible.
- **Related**: This is the "Category 1" case described in issue #146; listed here for completeness.

### Internal helper-column name collisions

<!-- machine-checked
operation: frame_aggregate, offset, rank, window_aggregation, aggregation, scalar_aggregate, binning
framework: pandas, polars_lazy, pyarrow, duckdb, sqlite
condition: a hardcoded helper-column name would overwrite or be shadowed by a same-named user column
mitigation_location:
- mloda/community/feature_groups/data_operations/helper_columns.py
regression_test:
- mloda/community/feature_groups/data_operations/tests/test_helper_columns.py::TestUniqueHelperName::test_returns_base_when_absent
- mloda/community/feature_groups/data_operations/tests/test_helper_columns.py::TestUniqueHelperName::test_first_collision_appends_suffix_1
- mloda/testing/feature_groups/data_operations/mixins/reserved_columns.py::ReservedColumnsTestMixin::test_mixin_reserved_column_collision_accepted
- mloda/testing/feature_groups/data_operations/mixins/reserved_columns.py::ReservedColumnsTestMixin::test_mixin_helper_column_name_collision_survives
-->

- **Operations**: every row-preserving op that tags rows with an internal helper column, for example `frame_aggregate`, `offset`, `rank`, `window_aggregation`, plus the mode helpers shared by `aggregation` / `scalar_aggregate` / `window_aggregation`, and the SQL `binning` NTILE path.
- **Where it lives**: the non-SQL backends use `unique_helper_name(base, taken)` in `mloda/community/feature_groups/data_operations/helper_columns.py`; the SQL backends (DuckDB, SQLite) use `pick_helper_column_name(taken=...)` from `mloda_plugins` `sql_utils`.
- **Reference behavior**: PyArrow's reference implementation does not need helper columns and is silent about column names starting with `__mloda_`.
- **Native framework behavior**: An implementation that adds an internal helper column (for example to record original row order before a reordering window function) under a hardcoded name would overwrite user data or be dropped silently if the input already carried a column with that name.
- **Mitigation kind**: Collision-free naming.
- **How**: Before adding a helper, each backend requests a name that is provably absent from the current frame. `unique_helper_name(base, taken)` returns `base` if absent, else the lowest `base_N` (N>=1) not in `taken`; `pick_helper_column_name(taken=set(data.columns) | {feature_name})` does the equivalent for the SQL relations. There is no reserved namespace and no `__mloda_` reject-guard: the earlier `assert_no_reserved_columns()` / `RESERVED_PREFIX` mechanism was removed in #221, so user columns of any name (including `__mloda_`-prefixed) are accepted by every backend.
- **Regression signal**: `mloda/community/feature_groups/data_operations/tests/test_helper_columns.py` unit-tests `unique_helper_name`. The shared `ReservedColumnsTestMixin` runs per framework and asserts that a `__mloda_`-prefixed user column is accepted (the call returns a non-null result) on every backend.

### SQLite lacks `percentile` and `reverse` (and the string ops above)

<!-- machine-checked
operation: percentile, string
framework: sqlite
condition: SQLite has no percentile implementation and no native reverse
mitigation_location:
- mloda/community/feature_groups/data_operations/string/sqlite_string.py
regression_test:
- mloda/community/feature_groups/data_operations/string/tests/test_sqlite.py::TestSqliteUnsupportedOps::test_unsupported_op_does_not_match
- mloda/community/feature_groups/data_operations/tests/test_framework_support_matrix.py::test_framework_support_matrix_is_in_sync
-->

- **Operations**: `row_preserving/percentile` and `string`.
- **Mitigation kind**: Excluded op.
- **How**: SQLite has no `percentile` test class at all (`row_preserving/percentile/tests/` ships no `test_sqlite.py`), and `string/tests/test_sqlite.py` uses `supported_ops()` to restrict the covered set. The missing percentile column is pinned by the framework-support-matrix drift check.
- **Related**: Category 1 of issue #146.

### SQLite and DuckDB time-frame use correlated subqueries (O(N^2) per partition)

<!-- machine-checked
operation: frame_aggregate
framework: sqlite, duckdb
condition: time-frame windows use a correlated subquery with an explicit same-ts peer tiebreaker
mitigation_location:
- mloda/community/feature_groups/data_operations/row_preserving/frame_aggregate/sqlite_frame_aggregate.py
- mloda/community/feature_groups/data_operations/row_preserving/frame_aggregate/duckdb_frame_aggregate.py
regression_test:
- mloda/testing/feature_groups/data_operations/row_preserving/frame_aggregate/frame_aggregate.py::FrameAggregateTestBase::test_cross_framework_time_window_with_same_ts_peers
-->

- **Operations**: `row_preserving/frame_aggregate` (only the `time` frame type).
- **Mitigation kind**: Accepted complexity (correctness preserved).
- **How**: Both backends issue a correlated subquery `SELECT agg(s.{col}) FROM t s WHERE s.{partition}=t.{partition} AND s.{ts} >= t.{ts} - delta AND (s.{ts} < t.{ts} OR (s.{ts}=t.{ts} AND s.{tiebreak}<=t.{tiebreak}))`. SQLite uses `julianday()` arithmetic (preserves sub-second precision; `datetime()` would truncate) with `rowid` as the peer tiebreaker. DuckDB uses `INTERVAL '{N}' {unit}` arithmetic with `ROW_NUMBER()` as the tiebreaker. The tiebreaker is required to match the PyArrow reference's positional `rows[:pos+1]` semantics on same-ts peers; SQL `RANGE` windows and `BETWEEN` predicates without a tiebreaker include all peers (both earlier and later in physical position).
- **Related**: parent #183, implementing #202.

### Polars time-frame adds 1ns-per-row offset to break same-ts peer ties

<!-- machine-checked
operation: frame_aggregate
framework: polars_lazy
condition: rolling_*_by is value-based and includes later same-ts peers unless offset apart
mitigation_location:
- mloda/community/feature_groups/data_operations/row_preserving/frame_aggregate/polars_lazy_frame_aggregate.py
regression_test:
- mloda/testing/feature_groups/data_operations/row_preserving/frame_aggregate/frame_aggregate.py::FrameAggregateTestBase::test_cross_framework_time_window_with_same_ts_peers
-->

- **Operations**: `row_preserving/frame_aggregate` (only the `time` frame type).
- **Mitigation kind**: Accepted complexity (correctness preserved).
- **How**: Polars `rolling_*_by` with `closed="both"` is value-based and includes every row whose `by` value equals the current row's value, even peers that come later in physical position. The PyArrow reference uses `rows[:pos+1]` after a stable sort, excluding later peers. `polars_lazy_frame_aggregate.py` casts the `order_by` column to `datetime[ns]` and adds `pl.duration(nanoseconds=row_index)` into a temporary `__mloda_synth_ts__` column, then runs `rolling_*_by` on the synthetic column. The window string is extended by `{N}ns` (where N is total row count) so a peer at the exact lower bound is not lost to the offset. The synthetic column is dropped before returning. See `polars_lazy_frame_aggregate.py`.
- **Related**: parent #183, implementing #202.

### SQLite + Pandas reject month/year time windows

<!-- machine-checked
operation: frame_aggregate
framework: sqlite, pandas
condition: month/year calendar units diverge from relativedelta (SQLite) or are unsupported (Pandas)
mitigation_location:
- mloda/community/feature_groups/data_operations/row_preserving/frame_aggregate/sqlite_frame_aggregate.py
- mloda/community/feature_groups/data_operations/row_preserving/frame_aggregate/pandas_frame_aggregate.py
regression_test:
- mloda/testing/feature_groups/data_operations/row_preserving/frame_aggregate/frame_aggregate.py::FrameAggregateTestBase::test_time_frame_match_rejected_when_unsupported
- mloda/testing/feature_groups/data_operations/row_preserving/frame_aggregate/frame_aggregate.py::FrameAggregateTestBase::test_time_frame_config_rejected_when_unsupported
-->

- **Operations**: `row_preserving/frame_aggregate` (only `time` frame type with `month`/`year` units).
- **Mitigation kind**: Excluded unit.
- **How**: SQLite's native `datetime(ts, '-N months')` uses day-of-month rollover (Mar 31 -1mo = Mar 3), diverging from `dateutil.relativedelta` (= Feb 28) used by the PyArrow reference. Pandas `.rolling(window="...", on=ts)` only accepts fixed-frequency offsets and has no native calendar-anchored option. Rather than fall back to a Python loop (which would defeat the point of running inside the engine), both `SqliteFrameAggregate.SUPPORTED_TIME_UNITS` and `PandasFrameAggregate.SUPPORTED_TIME_UNITS` exclude `month`/`year`, so features like `value__sum_1_month_window` are rejected at `match_feature_group_criteria` time. Both support `second`/`minute`/`hour`/`day`/`week`. Polars and DuckDB express month/year natively and remain ✓ for those units.
- **Related**: parent #183, implementing #202.

### All backends reject mask + `source_col == order_by` in time frames

<!-- machine-checked
operation: frame_aggregate
framework: pandas, polars_lazy, duckdb, sqlite, python_dict
condition: a mask with source_col == order_by in a time frame cannot be simulated natively
mitigation_location:
- mloda/community/feature_groups/data_operations/row_preserving/frame_aggregate/pandas_frame_aggregate.py
- mloda/community/feature_groups/data_operations/row_preserving/frame_aggregate/polars_lazy_frame_aggregate.py
- mloda/community/feature_groups/data_operations/row_preserving/frame_aggregate/duckdb_frame_aggregate.py
- mloda/community/feature_groups/data_operations/row_preserving/frame_aggregate/sqlite_frame_aggregate.py
- mloda/community/feature_groups/data_operations/row_preserving/frame_aggregate/python_dict_frame_aggregate.py
regression_test:
- mloda/testing/feature_groups/data_operations/row_preserving/frame_aggregate/frame_aggregate.py::FrameAggregateTestBase::test_time_window_source_equals_order_with_mask_rejected
-->

- **Operations**: `row_preserving/frame_aggregate` (only the `time` frame type, with `source_col == order_by` and a mask present).
- **Mitigation kind**: Excluded shape.
- **How**: The PyArrow reference applies the mask to `source_col` before computing the window. When `source_col == order_by`, mask-write clobbers the order column with null, and the reference's `current_order is None` branch returns just `[self]`. None of the native time-window primitives can simulate this: pandas `rolling(on=ts)` cannot; polars `rolling_*_by` uses the unmasked `order_by` for window bounds even when the masked source is a temp column; the DuckDB and SQLite correlated subqueries wrap only the aggregate expression in `CASE WHEN ... THEN source END`, leaving the bounds operating on the unmasked column. PythonDict builds `order_by` and the masked source as two independently-built Python lists, so it has the same coupling gap even though it is otherwise unconstrained by engine SQL/pandas limitations. Each backend raises a `ValueError` when this combo is detected at runtime instead of silently producing a wrong result. Non-time frames continue to work via a separate temp column (or list) for the masked source.
- **Related**: parent #183, implementing #202.

### Pandas / Polars-lazy native time-rolling rejects null `order_by`

<!-- machine-checked
operation: frame_aggregate
framework: pandas, polars_lazy
condition: native time-rolling errors on null order_by; both pre-check and raise a clear ValueError
mitigation_location:
- mloda/community/feature_groups/data_operations/row_preserving/frame_aggregate/pandas_frame_aggregate.py
- mloda/community/feature_groups/data_operations/row_preserving/frame_aggregate/polars_lazy_frame_aggregate.py
regression_test:
- mloda/testing/feature_groups/data_operations/row_preserving/frame_aggregate/frame_aggregate.py::FrameAggregateTestBase::test_cross_framework_time_window_with_null_cutoff
-->

- **Operations**: `row_preserving/frame_aggregate` (only the `time` frame type).
- **Mitigation kind**: Excluded test + explicit runtime error.
- **How**: `pandas.DataFrame.groupby(...).rolling(on=ts)` raises `"ts values must not have NaT"`; Polars `rolling_*_by` panics. Both `PandasFrameAggregate._compute_frame` and `PolarsLazyFrameAggregate._compute_frame` pre-check the `order_by` column for nulls when `frame_type == "time"` and raise a `ValueError` naming the framework and column, turning the cryptic native error into an explicit refusal. `FrameAggregateTestBase.supports_null_order_in_time_window()` defaults `True`; pandas + polars-lazy override to `False`, skipping `test_cross_framework_time_window_with_null_cutoff`. DuckDB and SQLite implement the reference behavior (window = `[self]`) and run the test.
- **Related**: parent #183, implementing #202.

### SQLite + Polars reject std/var/median frame aggregates at match time

<!-- machine-checked
operation: frame_aggregate
framework: sqlite, polars_lazy
condition: std/var/median frame aggregates rejected at match time (SQLite: all frame types; Polars: cumulative/expanding only)
mitigation_location:
- mloda/community/feature_groups/data_operations/row_preserving/frame_aggregate/base.py
- mloda/community/feature_groups/data_operations/row_preserving/frame_aggregate/sqlite_frame_aggregate.py
- mloda/community/feature_groups/data_operations/row_preserving/frame_aggregate/polars_lazy_frame_aggregate.py
regression_test:
- mloda/community/feature_groups/data_operations/row_preserving/frame_aggregate/tests/test_sqlite.py::TestSqliteFrameAggregate::test_mixin_capability_hook_rejects
- mloda/community/feature_groups/data_operations/row_preserving/frame_aggregate/tests/test_polars_lazy.py::TestPolarsLazyFrameAggregate::test_mixin_capability_hook_rejects
-->

- **Operations**: `row_preserving/frame_aggregate` (aggregation-type axis of the capability hook).
- **Mitigation kind**: Excluded agg type.
- **How**: SQLite has no native `STD`/`VAR`/`MEDIAN` window functions, so `SqliteFrameAggregate.supported_op_subtypes()` sources the supported set from `_SQLITE_AGG_FUNCS` (`sum`/`avg`/`count`/`min`/`max`), rejecting `std`/`var`/`median` for every frame type. Polars has no cumulative `cum_std`/`cum_var`/`cum_median`, so `PolarsLazyFrameAggregate.supported_op_subtypes()` returns `_CUMULATIVE_AGG_TYPES` for `cumulative`/`expanding` frames (excluding `std`/`var`/`median`) and `_ROLLING_AGG_TYPES` (the full set) for `rolling`/`time`. The base `supports_compute_framework` hook resolves the agg type from the parsed name or `aggregation_type` option and rejects unsupported combinations at match time, rather than failing later inside `_compute_frame`. Pandas and DuckDB inherit the base `None` (unrestricted) and support all eight agg types.
- **Related**: issue #296.

### PythonDict NaN partition keys merge into one group; min/max/percentile skip NaN

<!-- machine-checked
operation: percentile, rank, offset, ffill, ema, sessionization, resample, aggregation, window_aggregation
framework: python_dict
condition: a raw NaN partition-key value splits into its own singleton group, and NaN in a min/max/percentile input is not skipped, unlike the PyArrow reference
mitigation_location:
- mloda/community/feature_groups/data_operations/python_dict_helpers.py
- mloda/community/feature_groups/data_operations/row_preserving/percentile/python_dict_percentile.py
- mloda/community/feature_groups/data_operations/row_preserving/rank/python_dict_rank.py
- mloda/community/feature_groups/data_operations/row_changing/resample/python_dict_resample.py
- mloda/community/feature_groups/data_operations/row_preserving/sessionization/python_dict_sessionization.py
regression_test:
- mloda/community/feature_groups/data_operations/row_preserving/percentile/tests/test_python_dict.py::TestPythonDictNanPartitionKeyGrouping::test_nan_partition_keys_grouped_together_matches_pyarrow_oracle
- mloda/community/feature_groups/data_operations/row_preserving/percentile/tests/test_python_dict.py::TestPythonDictPercentileNanValueSkipped::test_percentile_of_skips_nan_directly
- mloda/community/feature_groups/data_operations/row_preserving/rank/tests/test_python_dict.py::TestPythonDictRankNanPartitionKeyGrouping::test_nan_partition_rows_share_one_group_continues_numbering
- mloda/community/feature_groups/data_operations/row_changing/resample/tests/test_python_dict.py::TestPythonDictMinMaxSkipsNan::test_min_skips_nan_matches_pyarrow_oracle
- mloda/community/feature_groups/data_operations/row_changing/resample/tests/test_python_dict.py::TestPythonDictNanPartitionKeyGrouping::test_nan_partition_rows_merge_into_one_bucket_matches_pyarrow_oracle
- mloda/community/feature_groups/data_operations/row_preserving/sessionization/tests/test_python_dict.py::TestPythonDictNanPartitionKeyGrouping::test_nan_partition_rows_share_one_session_matches_pyarrow_group_by
-->

- **Operations**: `percentile`, `rank`, `offset`, `ffill`, `ema`, `sessionization`, `resample`, `aggregation`, `window_aggregation` (every PythonDict backend that partitions rows or reduces a value list).
- **Where it lives**: `mloda/community/feature_groups/data_operations/python_dict_helpers.py` (`group_key_value`, `is_nan`, `nulls_last_sort_key`, `values_equal`, `reduce_agg`), consumed by every PythonDict backend under `row_preserving/` and `row_changing/resample/`.
- **Reference behavior**: PyArrow's `Table.group_by()` merges all NaN keys of a column into a single group, distinct from the null/None group. `pc.min`/`pc.max`/`pc.quantile` skip NaN like a missing value.
- **Native PythonDict behavior (unmitigated)**: `float('nan')` never equals another NaN, so a raw `tuple(col[i] for col in partition_cols)` group key splits every NaN-valued row into its own singleton group. Python's builtin `min()`/`max()` and `sorted()`-based percentile interpolation propagate NaN instead of skipping it.
- **Mitigation kind**: Implementation fix.
- **How**: Most PythonDict backends build their group key through `group_key_value`, which substitutes a shared sentinel for any NaN component so NaN-keyed rows land in one group; `reduce_agg`'s `min`/`max` branches and `_percentile_of` filter NaN out of the reduced value list before reducing, matching PyArrow's skip-NaN semantics. `python_dict_sessionization.py` is the exception: it never calls `group_key_value`. Instead it sorts raw partition-key values through `partition_sort_key` (which gives NaN and None each their own contiguous sort tier) and compares adjacent sorted rows with `values_equal` (NaN-safe equality), so two NaN-keyed rows still land in one contiguous, un-poisoned session without ever building a normalized key.
- **Regression signal**: The percentile/rank/resample tests cited above build a live PyArrow `Table.group_by()` (or a PyArrow-oracle comparison) and assert the PythonDict backend groups/reduces identically; removing `group_key_value` or the NaN filter from `reduce_agg`/`_percentile_of` fails those. The cited sessionization test instead exercises `partition_sort_key` and `values_equal`; it does not exercise `group_key_value`, so removing `group_key_value` does not fail it.

### PythonDict `median` propagates NaN, agreeing with DuckDB and Polars; only Pandas skips it

<!-- machine-checked
operation: aggregation, resample, window_aggregation, frame_aggregate
framework: python_dict
condition: reduce_agg median only filters None, not NaN; DuckDB MEDIAN and Polars .median() propagate NaN the same way, only Pandas skips it
mitigation_location:
- mloda/community/feature_groups/data_operations/python_dict_helpers.py
regression_test:
- mloda/community/feature_groups/data_operations/tests/test_python_dict_helpers.py::TestReduceAggMedianDoesNotSkipNanDocumentedDivergence::test_median_of_value_and_nan_returns_nan
-->

- **Operations**: `aggregation`, `resample`, `window_aggregation`, `frame_aggregate` (every PythonDict backend whose `median` agg type routes through `reduce_agg`).
- **Where it lives**: `mloda/community/feature_groups/data_operations/python_dict_helpers.py` (`reduce_agg`'s `median` branch).
- **Reference behavior**: There is no PyArrow `median` kernel. The test suite's own cross-framework reference, `ReferenceAggregation._median` in `mloda/testing/feature_groups/data_operations/aggregation/reference.py`, filters only `None`, not NaN, so `[1.0, nan]` reduces to `nan` there too.
- **Cross-framework check**: verified directly against each engine: Pandas' `Series.median()` defaults to `skipna=True` and returns `1.0` for `[1.0, nan]`. DuckDB's `MEDIAN(...)` and Polars' `.median()` both propagate NaN and return `nan` for the same input, matching PythonDict and the test reference. Of the four frameworks that implement `median`, three (DuckDB, Polars, PythonDict) already agree with each other and with the reference; Pandas is the outlier.
- **Native PythonDict behavior**: `reduce_agg`'s `non_null` list filters out `None` only; NaN reaches `statistics.median` unfiltered, e.g. `[1.0, nan]` -> `nan`.
- **Mitigation kind**: Accepted divergence (no mitigation attempted).
- **How**: Making PythonDict skip NaN in `median` would trade its current agreement with DuckDB, Polars, and the test reference for agreement with Pandas alone, a lateral move (not a reduction) in the number of divergent pairs, plus an added branch on a hot path. There is no PyArrow oracle to arbitrate which convention is "correct." If this is ever revisited, Pandas is the implementation to reconsider, not PythonDict.
- **Regression signal**: `test_median_of_value_and_nan_returns_nan` pins `reduce_agg("median", [1.0, float("nan")])` to `nan`; a future change that filters NaN like Pandas would flip this assertion to `1.0`.

### PythonDict `group_key_value` merges `0.0` and `-0.0` into one group

<!-- machine-checked
operation: aggregation, resample, window_aggregation, frame_aggregate, rank, sessionization, percentile
framework: python_dict
condition: group_key_value passes 0.0 and -0.0 through unchanged, and Python's float equality/hash merge them as dict keys
mitigation_location:
- mloda/community/feature_groups/data_operations/python_dict_helpers.py
regression_test:
- mloda/community/feature_groups/data_operations/tests/test_python_dict_helpers.py::TestGroupKeyValueMergesSignedZeroDocumentedDivergence::test_positive_and_negative_zero_collide_into_one_group
-->

- **Operations**: `aggregation`, `resample`, `window_aggregation`, `frame_aggregate`, `rank`, `sessionization`, `percentile` (every PythonDict backend that groups rows by a float-valued partition column).
- **Where it lives**: `mloda/community/feature_groups/data_operations/python_dict_helpers.py` (`group_key_value`).
- **Reference behavior**: PyArrow's `Table.group_by()` hashes a float key bitwise, so `0.0` and `-0.0` (distinct bit patterns) land in two separate groups (verified directly: grouping `[0.0, -0.0]` produces 2 groups).
- **Cross-framework check**: verified directly against each engine: Pandas' `.groupby()`, Polars' `.group_by()`, and DuckDB's `GROUP BY` all merge `0.0` and `-0.0` into a single group too, the same as PythonDict. PyArrow is the only one of the five frameworks that treats them as distinct; PythonDict's raw-dict-key grouping matches the four-out-of-five majority (Pandas, Polars, DuckDB, PythonDict).
- **Native PythonDict behavior**: `group_key_value` only special-cases NaN; `0.0` and `-0.0` pass through unchanged. Python's `float.__eq__` and `float.__hash__` both treat `0.0 == -0.0` as `True` with equal hashes, so a plain `dict`/`set` keyed by the raw value merges them into one group regardless of what `group_key_value` does.
- **Mitigation kind**: Accepted divergence (no mitigation attempted).
- **How**: A `group_key_value`-only fix would not even be internally consistent: `values_equal` (used by the sort-based sessionization grouping path) and `partition_sort_key` both treat `0.0 == -0.0` on purpose (see `test_negative_zero_equals_zero`, `test_negative_zero_and_zero_sort_together`), so PythonDict's own sort-keyed sessionization backend would keep merging them while its dict-keyed backends stopped, a new internal inconsistency. A real fix needs all three functions to become sign-aware (`(value, math.copysign(1.0, value))` instead of the raw value on every hot grouping path), and would only trade PythonDict's current agreement with Pandas/Polars/DuckDB for agreement with PyArrow alone, a lateral move in total cross-framework divergence for a sign-of-zero edge case that is exceedingly rare in real partition columns. The divergence is documented instead of mitigated.
- **Regression signal**: `test_positive_and_negative_zero_collide_into_one_group` pins that a `group_key_value`-keyed dict merges `0.0` and `-0.0` into a single group; a future change that makes `group_key_value` sign-aware would flip this assertion.

### PythonDict and SQLite rank tie None and NaN in order-value runs; PyArrow >= 25, DuckDB, and Polars rank them apart

<!-- machine-checked
operation: rank
framework: python_dict, sqlite, duckdb, pandas, polars_lazy
condition: order_values_equal (python_dict) and SQLite's own RANK()/DENSE_RANK() tie None and NaN into one rank run; PyArrow >= 25.0's pc.rank, DuckDB's RANK()/DENSE_RANK(), and Polars' .rank() rank NaN and NULL as distinct tiers
mitigation_location:
- mloda/community/feature_groups/data_operations/python_dict_helpers.py
- mloda/community/feature_groups/data_operations/row_preserving/rank/python_dict_rank.py
- mloda/testing/feature_groups/data_operations/row_preserving/rank/reference.py
regression_test:
- mloda/community/feature_groups/data_operations/row_preserving/rank/tests/test_python_dict.py::TestPythonDictRankMixedNoneAndNanTieRun::test_rank_ties_none_and_nan
- mloda/community/feature_groups/data_operations/row_preserving/rank/tests/test_python_dict.py::TestPythonDictRankMixedNoneAndNanTieRun::test_dense_rank_ties_none_and_nan
- mloda/community/feature_groups/data_operations/row_preserving/rank/tests/test_python_dict.py::TestPythonDictRankMixedNoneAndNanTieRun::test_percent_rank_ties_none_and_nan
- mloda/community/feature_groups/data_operations/row_preserving/rank/tests/test_pandas.py::TestPandasRankNoneAndNanOrderBy::test_rank_ties_none_and_nan
- mloda/community/feature_groups/data_operations/row_preserving/rank/tests/test_pandas.py::TestPandasRankNoneAndNanOrderBy::test_dense_rank_ties_none_and_nan
- mloda/community/feature_groups/data_operations/row_preserving/rank/tests/test_pandas.py::TestPandasRankNoneAndNanOrderBy::test_percent_rank_ties_none_and_nan
- mloda/community/feature_groups/data_operations/row_preserving/rank/tests/test_sqlite.py::TestSqliteRankNoneAndNanOrderBy::test_nan_is_stored_as_null
- mloda/community/feature_groups/data_operations/row_preserving/rank/tests/test_sqlite.py::TestSqliteRankNoneAndNanOrderBy::test_rank_ties_none_and_nan
- mloda/community/feature_groups/data_operations/row_preserving/rank/tests/test_sqlite.py::TestSqliteRankNoneAndNanOrderBy::test_dense_rank_ties_none_and_nan
- mloda/community/feature_groups/data_operations/row_preserving/rank/tests/test_sqlite.py::TestSqliteRankNoneAndNanOrderBy::test_percent_rank_ties_none_and_nan
- mloda/community/feature_groups/data_operations/row_preserving/rank/tests/test_duckdb.py::TestDuckdbRankNoneAndNanOrderBy::test_rank_ranks_none_and_nan_apart
- mloda/community/feature_groups/data_operations/row_preserving/rank/tests/test_duckdb.py::TestDuckdbRankNoneAndNanOrderBy::test_dense_rank_ranks_none_and_nan_apart
- mloda/community/feature_groups/data_operations/row_preserving/rank/tests/test_duckdb.py::TestDuckdbRankNoneAndNanOrderBy::test_percent_rank_ranks_none_and_nan_apart
- mloda/community/feature_groups/data_operations/row_preserving/rank/tests/test_polars_lazy.py::TestPolarsLazyRankNoneAndNanOrderBy::test_rank_ranks_none_and_nan_apart
- mloda/community/feature_groups/data_operations/row_preserving/rank/tests/test_polars_lazy.py::TestPolarsLazyRankNoneAndNanOrderBy::test_dense_rank_ranks_none_and_nan_apart
- mloda/community/feature_groups/data_operations/row_preserving/rank/tests/test_polars_lazy.py::TestPolarsLazyRankNoneAndNanOrderBy::test_percent_rank_ranks_none_and_nan_apart
- mloda/testing/feature_groups/data_operations/row_preserving/rank/tests/test_reference.py::TestReferenceRankNoneAndNanOrderBy::test_row_number_ranks_none_and_nan_apart
- mloda/testing/feature_groups/data_operations/row_preserving/rank/tests/test_reference.py::TestReferenceRankNoneAndNanOrderBy::test_rank_ranks_none_and_nan_apart
- mloda/testing/feature_groups/data_operations/row_preserving/rank/tests/test_reference.py::TestReferenceRankNoneAndNanOrderBy::test_dense_rank_ranks_none_and_nan_apart
- mloda/testing/feature_groups/data_operations/row_preserving/rank/tests/test_reference.py::TestReferenceRankNoneAndNanOrderBy::test_percent_rank_ranks_none_and_nan_apart
-->

- **Operations**: `rank` (`rank`, `dense_rank`, `percent_rank`; the same tiering also reaches `row_number`/`ntile_N`/`top_N`/`bottom_N`, but only these three collapse ties into one shared output value).
- **Where it lives**: `mloda/community/feature_groups/data_operations/python_dict_helpers.py` (`is_null_like`, `nulls_last_sort_key`, `order_values_equal`), consumed by `PythonDictRank._apply_rank`; SQLite and DuckDB inherit their behavior directly from their own native `RANK()`/`DENSE_RANK()`/`PERCENT_RANK()` window functions, with no mloda-side NaN handling at all. `ReferenceRank` uses its own local, independent NaN-safe tiering (not PythonDict's helpers), so it stays a valid cross-framework oracle rather than reproducing PythonDict's exact choice.
- **Reference behavior**: PyArrow < 25.0's `pc.rank` tied None and NaN into one rank tier under `null_placement="at_end"`. PyArrow >= 25.0 changed this: NaN is now always ranked as an ordinary, distinct value ahead of null, under any `null_placement` (no parameter restores the old tie; verified directly against pyarrow 25.0.1). DuckDB's `RANK() ... NULLS LAST` independently ranks NaN and NULL apart too, so PyArrow's move brought it in line with DuckDB, not the other way around. `ReferenceRank` (built on `PyArrowTable`) now mirrors this: real values first, then NaN, then None, each its own tier.
- **Cross-framework check**: verified directly against each engine's `_compute_rank` on `order_by = [None, nan, None, 1.0]`: PythonDict and Pandas' own `.rank()` give `[2, 2, 2, 1]` (None and NaN tied) by genuine choice; SQLite's native `RANK()` gives the same `[2, 2, 2, 1]`, but trivially, since the sqlite3 driver stores any NaN REAL as SQL `NULL` on ingest, so its `RANK()` never sees a NaN to rank apart from `NULL` in the first place. DuckDB's `RANK()`/`DENSE_RANK()`, Polars' `.rank()`, and PyArrow 25.0.1's `pc.rank` all give `[3, 2, 3, 1]` (None and NaN ranked apart, NaN ordered as the larger value); `ReferenceRank` gives the same `[3, 2, 3, 1]` by matching PyArrow's own tiering. The real split is two groups: engines that can represent NaN and NULL as distinct values (DuckDB, Polars, PyArrow, and `ReferenceRank` by choice) rank them apart; engines that cannot keep the two distinct (pandas' float64 column has no separate NaN/null representation; SQLite coerces NaN to NULL at the driver level) tie them trivially. PythonDict is the only backend that can represent both and chooses to tie them anyway.
- **Mitigation kind**: Accepted divergence (no mitigation attempted).
- **How**: `nulls_last_sort_key` and `order_values_equal` fold NaN and `None` into one null-like tier (`is_null_like`) for order-value comparisons; this was added specifically to fix a tie-run-splitting bug (a NaN row could otherwise sort between two None rows in the same tier and break the tie run), predates and is unrelated to the PyArrow 25 bump. PyArrow 25, DuckDB, and Polars now disagree with that choice; Pandas and SQLite agree with it, though neither can actually represent the distinction PythonDict is choosing to collapse. Splitting None from NaN in PythonDict would trade agreement with Pandas for agreement with PyArrow 25/DuckDB/Polars, a lateral move, and would reopen the tie-run-splitting bug the current behavior exists to fix. The divergence is documented instead of mitigated.
- **Regression signal**: each backend's own test class above pins `order_by = [None, nan, None, 1.0]` to its verified `rank`/`dense_rank`/`percent_rank` output; a future change to any backend's null/NaN handling that flips its tie behavior fails its own pinned test. `ReferenceRank` (the canonical cross-framework reference `RankTestBase` uses elsewhere) previously sorted NaN as an ordinary value and compared ties with plain `==`, an infinite loop since `nan == nan` is `False`; it now uses a small local NaN-safe tiering that ranks NaN apart from None (matching its own PyArrow engine) so it no longer hangs and stays independent of `PythonDictRank`'s tie choice. `RankTestBase`'s canonical fixture still does not include NaN in `order_by`, since the five backends' genuine disagreement here means no single expected value could be asserted through the shared `test_cross_framework` comparison.

---

## Audit coverage (2026-05-28)

The full audit covered all twelve data operations: `binning`, `datetime`, `frame_aggregate`, `offset`, `percentile`, `point_arithmetic`, `rank`, `scalar_aggregate`, `scalar_arithmetic`, `window_aggregation`, `aggregation`, `string`. Every implementation file and every `*TestBase` was read.

| Operation | Frameworks audited | New divergence found? |
|---|---|---|
| aggregation | PyArrow, Pandas, Polars lazy, DuckDB, SQLite | No (all mitigated above) |
| binning | PyArrow, Pandas, Polars lazy, DuckDB, SQLite | No |
| datetime | PyArrow, Pandas, Polars lazy, DuckDB, SQLite | No |
| frame_aggregate | Pandas, Polars lazy, DuckDB, SQLite | No |
| offset | Pandas, Polars lazy, DuckDB, SQLite | No |
| percentile | Pandas, Polars lazy, DuckDB | No (float tolerance already accepted) |
| point_arithmetic | PyArrow, Pandas, Polars lazy, DuckDB, SQLite | Yes — SQLite divide-by-zero returns NULL instead of inf/nan (documented as accepted divergence above) |
| rank | Pandas, Polars lazy, DuckDB, SQLite | No (all mitigated above) |
| scalar_aggregate | PyArrow, Pandas, Polars lazy, DuckDB, SQLite | No |
| scalar_arithmetic | PyArrow, Pandas, Polars lazy, DuckDB, SQLite | No (PyArrow int÷int truncation mitigated by explicit float cast) |
| string | PyArrow, Pandas, Polars lazy, DuckDB, SQLite | No (SQLite ASCII mitigated) |
| window_aggregation | Pandas, Polars lazy, DuckDB, SQLite | No (all mitigated above) |

One unmitigated divergence is documented above (SQLite `point_arithmetic` divide-by-zero), with a per-backend test hook (`divide_zero_propagates_inf()`) so the cross-framework regression guard pins the expected behavior on each backend. The `expected_*()` hooks defined on `StringTestBase` (`expected_upper`, `expected_lower`, `expected_trim`, `expected_length`, `expected_reverse`) are present for future use but are not currently overridden by any framework: after #147, SQLite no longer matches the unicode-unsafe ops instead of returning a divergent result.

---

## When to add to this page

Add a new entry here if and only if all three hold:

1. A framework operator produces a measurably different result from PyArrow on a realistic input.
2. That difference cannot be hidden by the cross-framework comparison (i.e. it would require an `expected_*()` override, a `use_approx=True` bump, a `pytest.skip`, or a `supported_ops()` exclusion).
3. The decision (fix vs. document vs. exclude) has been made and landed in code.

Do not add speculative entries. If an audit only uncovered a hypothetical divergence, add a failing regression test first so the entry corresponds to something the test suite measures.

Every entry above carries a `<!-- machine-checked ... -->` block right under its heading, with `operation`, `framework`, `condition`, `mitigation_location`, and `regression_test` fields. This block is the authoritative, machine-checked record of each entry; the surrounding prose is explanatory and is not itself verified, so when the two could disagree, trust the block. `mloda/community/feature_groups/data_operations/tests/test_known_divergences.py` runs in `tox` and asserts (a) every entry has such a block with all fields populated, (b) every `regression_test` reference resolves to a real, importable test, (c) every `mitigation_location` path exists, and (d) every `operation` / `framework` token is a name the framework-support-matrix check recognizes. A new entry without a resolvable guard fails the build, so the doc cannot rot silently.

Format rules for the block: `operation`, `framework`, and `condition` are single-line values; `mitigation_location` and `regression_test` are `- ` lists, one item per line. Use repo-relative paths and `path.py::Class[::method]` (or `path.py::test_function`) for the test reference. To reference a guard that lives on a `*TestBase` class, name a specific `::method` (pytest collects base classes only through their per-framework subclasses, so a bare base-class reference is rejected).

---

## Related

- [Framework support matrix](framework-support-matrix.md) - The full operation x framework capability table; every divergence-driven exclusion on this page corresponds to one or more matrix cells.
- [Reference implementation pattern](03-reference-implementation.md) - Why PyArrow is authoritative.
- [Supported ops per framework](04-supported-ops.md) - The exclusion mechanism used when a framework cannot match PyArrow.
- [Row-preserving contract](02-row-preserving-contract.md) - The invariant every row-preserving op must honor.
