# Reference Implementation Pattern

Every data operation has one framework designated as the reference. Other frameworks are correct if and only if they match the reference on the shared test suite. PyArrow is the reference for the operations that ship today.

**What**: One framework implementation is the source of truth. Cross-framework tests run the same feature through both the target framework and the reference, then assert the two results match.
**When**: Every time you add a new operation or add a new framework implementation for an existing operation.
**Why**: Without a fixed reference, frameworks drift into mutually incompatible results (e.g. different NULL handling, different tie-breaking in ranks). Making PyArrow authoritative keeps semantics aligned without requiring everyone to agree on an abstract spec.
**Where**: `mloda/testing/feature_groups/data_operations/base.py::DataOpsTestBase._compare_with_reference`.
**How**: Each framework mixin provides `create_test_data` (converts the canonical PyArrow table to the framework's native format) and `extract_column` (materializes the result back to a Python list for comparison).

---

## Why PyArrow

PyArrow implementations are:

- **Columnar and index-based**, so row order is trivially preserved.
- **Explicit about NULL** via `pa.array` masks, so NULL handling is unambiguous.
- **Deterministic**: `pc.sort_indices`, `pc.rank`, `pc.round` give the same result every run.

These properties make PyArrow a reasonable ground truth. It is not faster than the others; it is the cleanest semantic reference.

---

## How the comparison works

`_compare_with_reference` runs both implementations on equivalent data:

```python
# self.test_data is the framework-native test fixture
# self._arrow_table is the same data as a PyArrow table
fs = make_feature_set(feature_name, partition_by=partition_by, order_by=order_by)
result = self.implementation_class().calculate_feature(self.test_data, fs)
ref    = self.reference_implementation_class().calculate_feature(self._arrow_table, fs)

result_col = self.extract_column(result, feature_name)
ref_col    = _extract_column(ref, feature_name)
assert result_col == ref_col  # or pytest.approx for floats
```

`self.test_data` is produced by the framework mixin's `create_test_data(pa.Table)`. The canonical input is always PyArrow; the mixin converts it once per test. This keeps the fixtures aligned: both sides compute from the *same* starting values, just in different native representations.

---

## What "matching" means

For integer operations, matching means exact element-wise equality including NULLs. For floating-point operations, the test harness accepts `pytest.approx` with `rel=1e-6` when `use_approx=True`. That tolerance exists because frameworks differ in accumulation order (running sums through SQL engines vs. columnar reductions), not because results are allowed to drift.

If your framework produces NaN where PyArrow produces NULL, the comparison fails. NaN and NULL are not interchangeable; frame-level tests treat them as distinct.

---

## Canonical test data

Row-preserving and aggregation tests share a canonical input: 12 rows with columns `region`, `category`, `value_int`, `value_float`, `ts`, `name`. Each operation's test base asserts specific expected values against this fixture. When you subclass the test base for a new framework, you do not redefine the data or the expected values. You only implement the adapter methods that convert between PyArrow and your framework's native type.

---

## When the reference itself is wrong

Bugs happen. If the PyArrow implementation is wrong:

1. Fix the PyArrow implementation.
2. Update the operation's test base (`BinningTestBase`, `AggregationTestBase`, etc.) so the expected values reflect the correct result.
3. Run every framework's test class. Every non-PyArrow implementation that was "matching" the bug will now fail.
4. Fix each framework implementation so it matches the corrected reference.

Do not "fix" a non-PyArrow framework to agree with the PyArrow bug. The reference is where correctness lives, so fixing the reference is what realigns the ecosystem.

---

## Related

- [Row-preserving contract](02-row-preserving-contract.md) - The row-count-and-order invariant that the reference comparison enforces.
- [Supported ops per framework](04-supported-ops.md) - How a framework skips comparisons for ops it cannot express.
- [Known divergences](known-divergences.md) - Audited cases where a framework's native operator would diverge from PyArrow, and the fix or exclusion used.
- [Adding a new data operation](10-adding-new-operation.md) - Where to put the PyArrow reference when adding an op.
