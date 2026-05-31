# Row-Preserving Contract

Row-preserving operations must return an output whose row count and row order match the input. This invariant is what lets users chain analytic transforms without surprise reshuffles.

**What**: For every row-preserving operation, `len(output) == len(input)` and row *i* in the output corresponds to row *i* in the input.
**When**: Applies to every feature group under `data_operations/row_preserving/`: binning, window aggregation, rank, offset, percentile, scalar aggregate, scalar arithmetic, point arithmetic, frame aggregate, datetime.
**Why**: Row-preserving ops broadcast a computed value back onto each row. If the framework reorders rows (some SQL window functions do), downstream joins on index position break silently.
**Where**: The contract is enforced by cross-framework tests. Row count and row order are both asserted in `DataOpsTestBase._compare_with_reference`.
**How**: Each framework implementation must return rows in the same order they came in. If the native operator reorders (DuckDB `NTILE`), the implementation records original positions and restores them.

---

## What the contract asserts

`mloda/testing/feature_groups/data_operations/base.py` contains `_compare_with_reference`, which is the shared harness every row-preserving test uses:

```python
# Paraphrased from DataOpsTestBase
result = self.implementation_class().calculate_feature(self.test_data, fs)
ref    = self.reference_implementation_class().calculate_feature(self._arrow_table, fs)

result_col = self.extract_column(result, feature_name)
ref_col    = _extract_column(ref, feature_name)

assert len(result_col) == len(ref_col), f"row count mismatch"
assert result_col == ref_col            # exact match (or pytest.approx for floats)
```

Two assertions, one invariant:

1. **Row count**: `len(result_col) == len(ref_col)`.
2. **Row order**: element-wise equality against the reference (PyArrow). Non-matching row order fails the second assertion even if the multiset of values is identical.

---

## Per-framework notes

| Framework | Default behavior | Workaround needed? |
|---|---|---|
| PyArrow | Operates on columnar arrays by index; naturally preserves row order | No |
| Pandas | Index-aligned assignments (`df[col] = ...`) preserve order | No, as long as you avoid `groupby(...).apply()` patterns that reset index |
| Polars (lazy) | `pl.when/then/otherwise` and `over(...)` preserve order | No |
| SQLite | SQL result order is undefined unless `ORDER BY` is specified, but the implementations re-select columns rather than applying `ORDER BY` | No, but see the positional-append note below |
| DuckDB | Most operators preserve order. `NTILE()` reorders by its `ORDER BY` clause | Yes, see below |

---

## The SQLite positional-append assumption

The SQLite arithmetic backends (`scalar_arithmetic`, `point_arithmetic`) compute the new column with a bare `SELECT (expr) AS feat FROM <table>` and then call `data.append_column(feature_name, result_values)`. `append_column` aligns `result_values` to the existing rows **by position**, so correctness depends on the unordered `SELECT` returning rows in the relation's stored order. SQLite does return rows in `rowid` order for such a simple scan, which matches the stored order, so the invariant holds today.

The dependency is implicit, so each `_compute_arithmetic` carries a comment stating it: the `SELECT` must stay free of `ORDER BY`, `JOIN`, `GROUP BY`, or `DISTINCT`. Adding any of those could reorder rows and silently misalign the appended column. By contrast the DuckDB backend stays in a single relation via `data.project("*, (expr) AS feat")`, so it has no equivalent positional assumption.

---

## The DuckDB NTILE workaround

DuckDB's `NTILE()` window function requires an `ORDER BY` clause and emits rows in that order, not in input order. PyArrow's rank-based `qbin` assigns labels by index and preserves the original sequence. To make the two match, DuckDB tags each input row with a row-number column before the window runs, then re-sorts on that column to restore the original sequence.

The SQL plugins use the typed `DuckdbRelation` / `SqliteRelation` API (`with_row_number`, `window`) rather than hand-written `OVER (...)` strings, and choose a collision-free helper-column name at runtime with `pick_helper_column_name`. From `mloda/community/feature_groups/data_operations/row_preserving/binning/duckdb_binning.py`:

```python
# PyArrow parity: PyArrow _quantile_bin() assigns bins via index
# mapping and naturally preserves row order. DuckDB NTILE()
# reorders rows via ORDER BY; tag positions with a row-number
# column and restore via .order() to match PyArrow output.
rn = pick_helper_column_name(taken=set(data.columns) | {feature_name})
tagged     = data.with_row_number(rn)
with_qbin  = tagged.window(f"NTILE({n_bins})", qbin_ntile, partition_by=[qbin_part], order_by=[source_col])
sorted_rel = with_qbin.order(quote_ident(rn))
# drop the helper columns from the final projection
```

This pattern generalizes. Any time a framework's native operator reorders rows, the implementation must record positions before the operator runs and restore them afterward.

Every backend now picks provably collision-free helper names at runtime, so user columns of any name (including the `__mloda_` prefix) are accepted. The SQL backends (DuckDB, SQLite) use `pick_helper_column_name` from `mloda_plugins` `sql_utils`; the pandas, polars and pyarrow backends use `unique_helper_name(base, taken)` from `mloda/community/feature_groups/data_operations/helper_columns.py`. The old `__mloda_` reject-guard (`assert_no_reserved_columns()`) was removed in #221, so no reserved namespace exists. See [Known divergences](known-divergences.md) for the full entry.

---

## How to verify a new implementation honors the contract

Write the test by inheriting from the operation's test base and the relevant framework mixin. You do not need to add row-count assertions yourself; they are inherited from `DataOpsTestBase`.

```python
from mloda.testing.feature_groups.data_operations.mixins.duckdb import DuckdbTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.binning.binning import (
    BinningTestBase,
)
from mloda.community.feature_groups.data_operations.row_preserving.binning.duckdb_binning import (
    DuckdbBinning,
)


class TestDuckdbBinning(DuckdbTestMixin, BinningTestBase):
    @classmethod
    def implementation_class(cls):
        return DuckdbBinning
```

If your framework reorders, you will see row-count assertions pass but value comparisons fail with "row 0: 3 != reference 0". That means order is wrong, not math. Add the row-number workaround and try again.

---

## Aggregation is different

Aggregations under `data_operations/aggregation/` are **not** row-preserving. They reduce to one row per partition key. The [reference implementation](03-reference-implementation.md) still applies (PyArrow defines the expected result), but row count is expected to shrink.

---

## Related

- [Reference implementation pattern](03-reference-implementation.md) - Why PyArrow is the source of truth.
- [Binning](05-binning.md) - Concrete walkthrough of the DuckDB workaround.
- [Adding a new data operation](10-adding-new-operation.md) - How to wire up the tests that enforce this contract.
