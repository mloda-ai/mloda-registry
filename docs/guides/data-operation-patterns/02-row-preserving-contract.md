# Row-Preserving Contract

Row-preserving operations must return an output whose row count and row order match the input. This invariant is what lets users chain analytic transforms without surprise reshuffles.

**What**: For every row-preserving operation, `len(output) == len(input)` and row *i* in the output corresponds to row *i* in the input.
**When**: Applies to every feature group or operation group under `data_operations/row_preserving/`: binning, window aggregation, rank, offset, percentile, scalar aggregate, scalar arithmetic, point arithmetic, frame aggregate, datetime, time bucketization, ffill, ema, sessionization, lookup.
**Why**: Row-preserving ops broadcast a computed value back onto each row. If the framework reorders rows (some SQL window functions do), downstream joins on index position break silently.
**Where**: The contract is enforced by cross-framework tests. Row count and row order are both asserted in `DataOpsTestBase._compare_with_reference`.
**How**: Each framework implementation must return rows in the same order they came in. If the native operator reorders (DuckDB `NTILE` and, in DuckDB 2, partitioned windows), the implementation records original positions and restores them.

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
| SQLite | SQL result order is undefined unless `ORDER BY` is specified, but the implementations re-select columns rather than applying `ORDER BY` | No, but see the positional-append note below |
| DuckDB | Most operators preserve order. `NTILE()` reorders by its `ORDER BY` clause, and partitioned windows reorder too (DuckDB 2), restored the same way | Yes, see below |

---

## The SQLite positional-append assumption

The SQLite arithmetic backends (`scalar_arithmetic`, `point_arithmetic`) compute the new column with a bare `SELECT (expr) AS feat FROM <table>` and then call `data.append_column(feature_name, result_values)`. `append_column` aligns `result_values` to the existing rows **by position**, so correctness depends on the unordered `SELECT` returning rows in the relation's stored order. SQLite does return rows in `rowid` order for such a simple scan, which matches the stored order, so the invariant holds today.

The dependency is implicit, so each `_compute_arithmetic` carries a comment stating it: the `SELECT` must stay free of `ORDER BY`, `JOIN`, `GROUP BY`, or `DISTINCT`. Adding any of those could reorder rows and silently misalign the appended column. By contrast the DuckDB backend stays in a single relation via `data.project("*, (expr) AS feat")`, so it has no equivalent positional assumption.

---

## The DuckDB NTILE workaround

DuckDB's `NTILE()` window function requires an `ORDER BY` clause and emits rows in that order, not in input order. PyArrow's rank-based `qbin` assigns labels by index and preserves the original sequence. To make the two match, DuckDB wraps the window in `preserve_row_order` (in `mloda/community/feature_groups/data_operations/duckdb_helpers.py`), which tags each input row with a row-number column, runs the callback, re-sorts on the tag to restore the original sequence, and drops it.

The SQL plugins use the typed `DuckdbRelation` / `SqliteRelation` API (`with_row_number`, `window`) rather than hand-written `OVER (...)` strings, and choose a collision-free helper-column name at runtime with `pick_helper_column_name`. Inside the callback, `rel.columns` already holds the tag, so helper picks against it cannot collide. Paraphrased from `mloda/community/feature_groups/data_operations/row_preserving/binning/duckdb_binning.py`:

```py
# PyArrow parity: results come back in original row order.
def compute(rel, rn):
    qbin_part = pick_helper_column_name(taken=set(rel.columns) | {feature_name})
    qbin_ntile = pick_helper_column_name(taken=set(rel.columns) | {feature_name, qbin_part})
    rel = rel.project(f"*, {part_case} AS {quote_ident(qbin_part)}")
    rel = rel.window(f"NTILE({n_bins})", qbin_ntile, partition_by=[qbin_part], order_by=[source_col])
    # project out the helper columns and add the bin label
    return rel.project(f"{keep}, {expr} AS {quoted_feature}")


return preserve_row_order(data, feature_name, compute)
```

This pattern generalizes. Any time a framework's native operator reorders rows, the implementation must record positions before the operator runs and restore them afterward.

Every backend now picks provably collision-free helper names at runtime, so user columns of any name (including the `__mloda_` prefix) are accepted. The SQL backends (DuckDB, SQLite) use `pick_helper_column_name` from `mloda_plugins` `sql_utils`; the pandas, polars and pyarrow backends use `unique_helper_name(base, taken)` from `mloda/community/feature_groups/data_operations/helper_columns.py`. See [Known divergences](known-divergences.md) for the full entry.

---

## How to verify a new implementation honors the contract

Inherit the operation's test base and the framework mixin (template in [Adding a new data operation, Step 6](10-adding-new-operation.md#step-6-wire-up-the-framework-test-classes)); the row-count and order assertions come from `DataOpsTestBase`.

If your framework reorders, you will see row-count assertions pass but value comparisons fail with "row 0: 3 != reference 0". That means order is wrong, not math. Add the row-number workaround (on DuckDB, wrap the reordering step in `preserve_row_order`) and try again.

---

## Aggregation is different

Aggregations under `data_operations/aggregation/` are **not** row-preserving. They reduce to one row per partition key. The [reference implementation](03-reference-implementation.md) still applies (a pure Python reference defines the expected result, through the function `reference_aggregation(table, features)`), but row count is expected to shrink.

---

## Related

- [Reference implementation pattern](03-reference-implementation.md) - Which reference each operation uses (PyArrow or pure Python).
- [Binning](05-binning.md) - `bin` vs `qbin` and the NTILE equivalence.
- [Adding a new data operation](10-adding-new-operation.md) - How to wire up the tests that enforce this contract.
