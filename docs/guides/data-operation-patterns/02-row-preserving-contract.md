# Row-Preserving Contract

Row-preserving operations must return an output whose row count and row order match the input. This invariant is what lets users chain analytic transforms without surprise reshuffles.

**What**: For every row-preserving operation, `len(output) == len(input)` and row *i* in the output corresponds to row *i* in the input.
**When**: Applies to every feature group or operation group under `data_operations/row_preserving/`: binning, window aggregation, rank, offset, percentile, scalar aggregate, scalar arithmetic, point arithmetic, frame aggregate, datetime, time bucketization, ffill, ema, sessionization, lookup.
**Why**: Row-preserving ops broadcast a computed value back onto each row. If the framework reorders rows (some SQL window functions do), downstream joins on index position break silently.
**Where**: The contract is enforced by cross-framework tests. Row count and row order are both asserted in `DataOpsTestBase._compare_with_reference`.
**How**: Each framework implementation must return rows in the same order they came in. On SQL backends `window()` and `with_row_number()` keep input order since mloda 0.19; a step that reorders outside `window()` records original positions and restores them.

---

## What the contract asserts

Row count and row order are both asserted by `DataOpsTestBase._compare_with_reference` (element-wise equality against the reference, so a reordered result fails even when the value multiset matches); see [How the comparison works](03-reference-implementation.md#how-the-comparison-works).

---

## Per-framework notes

| Framework | Default behavior | Workaround needed? |
|---|---|---|
| PyArrow | Operates on columnar arrays by index; naturally preserves row order | No |
| Pandas | Index-aligned assignments (`df[col] = ...`) preserve order | No, as long as you avoid `groupby(...).apply()` patterns that reset index |
| Polars (lazy) | `pl.when/then/otherwise` and `over(...)` preserve order | No |
| SQLite | `window()` keeps input order since mloda 0.19; fetch-and-append paths scan without `ORDER BY`, the order `append_column` aligns by | No, but see the positional-append note below |
| DuckDB | `window()` and `with_row_number()` keep input order since mloda 0.19; raw SQL and joins may reorder | Only for non-window reordering steps, see below |

---

## The SQLite positional-append assumption

The SQLite fetch-and-append backends (for example `scalar_arithmetic`, `point_arithmetic`) compute the new column with a bare `SELECT (expr) AS feat FROM <table>` and then call `data.append_column(feature_name, result_values)`. `append_column` aligns `result_values` to the existing rows **by position**, so correctness depends on the unordered `SELECT` returning rows in the relation's stored order. A plain scan returns rows in that same order, so the invariant holds. Do not add `ORDER BY rowid`: views have no `rowid`, and an op's input is a view after the first feature.

The dependency is implicit, so each `_compute_arithmetic` carries a comment stating it: the `SELECT` must stay free of `ORDER BY`, `JOIN`, `GROUP BY`, or `DISTINCT`. Adding any of those could reorder rows and silently misalign the appended column. By contrast the DuckDB backend stays in a single relation via `data.project("*, (expr) AS feat")`, so it has no equivalent positional assumption.

---

## SQL backends: window() keeps row order

Since mloda 0.19, `window()` and `with_row_number()` on `DuckdbRelation` and `SqliteRelation` keep input row order, so window-only code needs no workaround.

A step that reorders outside `window()` (raw SQL via `rel.query`, joins) must record positions and restore them; `preserve_row_order` from `mloda/community/feature_groups/data_operations/duckdb_helpers.py` is the helper (lookup does it by hand). It tags each input row with a row-number column, runs the callback, re-sorts on the tag and drops it. When a window needs the input position as a tie-break, `with_input_position` tags without re-sorting. Paraphrased from `_compute_time_frame` in `mloda/community/feature_groups/data_operations/row_preserving/frame_aggregate/duckdb_frame_aggregate.py`:

```py
def compute(tagged, rn):
    return tagged.query("tagged", sql_using_rn)


return preserve_row_order(data, feature_name, compute)
```

The SQL plugins use the typed `DuckdbRelation` / `SqliteRelation` API (`with_row_number`, `window`) rather than hand-written `OVER (...)` strings, and choose a collision-free helper-column name at runtime with `pick_helper_column_name`. Inside a tagged callback, `rel.columns` already holds the tag, so helper picks against it cannot collide.

Every backend now picks provably collision-free helper names at runtime, so user columns of any name (including the `__mloda_` prefix) are accepted. The SQL backends (DuckDB, SQLite) use `pick_helper_column_name` from `mloda_plugins` `sql_utils`; the pandas, polars and pyarrow backends use `unique_helper_name(base, taken)` from `mloda/community/feature_groups/data_operations/helper_columns.py`. See [Known divergences](known-divergences.md) for the full entry.

---

## How to verify a new implementation honors the contract

Inherit the operation's test base and the framework mixin (template in [Adding a new data operation, Step 6](10-adding-new-operation.md#step-6-wire-up-the-framework-test-classes)); the row-count and order assertions come from `DataOpsTestBase`.

If your framework reorders, you will see row-count assertions pass but value comparisons fail with "row 0: 3 != reference 0". That means order is wrong, not math. Add the row-number workaround (on DuckDB, wrap a non-window reordering step in `preserve_row_order`) and try again.

---

## Aggregation is different

Aggregations under `data_operations/aggregation/` are **not** row-preserving. They reduce to one row per partition key. The [reference implementation](03-reference-implementation.md) still applies (a pure Python reference defines the expected result, through the function `reference_aggregation(table, features)`), but row count is expected to shrink.

---

## Related

- [Reference implementation pattern](03-reference-implementation.md) - Which reference each operation uses (PyArrow or pure Python).
- [Binning](05-binning.md) - `bin` vs `qbin` and the NTILE equivalence.
- [Adding a new data operation](10-adding-new-operation.md) - How to wire up the tests that enforce this contract.
