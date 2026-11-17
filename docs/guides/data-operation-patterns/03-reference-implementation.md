# Reference Implementation Pattern

Every data operation has one framework designated as the reference. Other frameworks are correct if and only if they match the reference on the shared test suite. The aggregating operations (aggregation, window aggregation, frame aggregate, scalar aggregate) and point and scalar arithmetic have a pure Python reference. PyArrow is the reference for most other operations that ship today; where it has no exact grouped kernel, a pure-Python reference applies PyArrow's semantics (see [When PyArrow has no kernel](#when-pyarrow-has-no-kernel-or-is-the-outlier)).

**What**: One framework implementation is the source of truth. Cross-framework tests run the same feature through both the target framework and the reference, then assert the two results match.
**When**: Every time you add a new operation or add a new framework implementation for an existing operation.
**Why**: Without a fixed reference, frameworks drift into mutually incompatible results (e.g. different NULL handling, different tie-breaking in ranks). For the operations that keep a PyArrow-based reference, making PyArrow authoritative keeps semantics aligned without requiring everyone to agree on an abstract spec; the aggregating operations pin the same idea in a written contract instead.
**Where**: `mloda/testing/feature_groups/data_operations/base.py::DataOpsTestBase._compare_with_reference`.
**How**: Each framework mixin provides `create_test_data` (converts typed test data to the framework's native format) and `extract_column` (materializes the result back to a Python list for comparison).

---

## Why PyArrow

For the families that keep a PyArrow-based reference, PyArrow implementations are:

- **Columnar and index-based**, so row order is trivially preserved.
- **Explicit about NULL** via `pa.array` masks, so NULL handling is unambiguous.
- **Deterministic**: `pc.sort_indices`, `pc.rank`, `pc.round` give the same result every run.

These properties make PyArrow a reasonable ground truth for those families. It is not faster than the others; it is the cleanest semantic reference.

---

## How the comparison works

`_compare_with_reference` runs both implementations on equivalent data:

```py
# self.test_data is the framework-native test fixture
# self._arrow_table is the same data as a PyArrow table
fs = make_feature_set(feature_name, partition_by=partition_by, order_by=order_by)
result = calculate_with(self.implementation_class(), self.test_data, fs)
ref = self.reference_calculate(self._arrow_table, fs)

result_col = self.extract_column(result, feature_name)
ref_col = _extract_column(ref, feature_name)
assert result_col == ref_col  # or pytest.approx for floats
```

Aggregation, window aggregation, frame aggregate, scalar aggregate, point arithmetic, scalar arithmetic, rank, offset and percentile are the exceptions: each is one `OperationFeatureGroup` served by one `Implementation` per framework, so their test bases have no `reference_implementation_class`. Their references are the functions `reference_aggregation`, `reference_window_aggregation`, `reference_frame_aggregate`, `reference_scalar_aggregate`, `reference_point_arithmetic`, `reference_scalar_arithmetic`, `reference_rank`, `reference_offset` and `reference_percentile` (each `(group, table, features)`, in the operation's `reference.py` under `mloda/testing/feature_groups/data_operations/`; the two arithmetic references live in the shared `row_preserving/arithmetic/reference.py`), which run the parsing loop of the group under test with the kit's compute kernel and import nothing from the code they judge; `reference_calculate` passes `feature_group_class()` and reaches them, and the test kit runs the implementation with `calculate_with` (`mloda/testing/feature_groups/data_operations/helpers.py`). Their NaN divergences come from the implementation's shipped `divergences` records. Lookup, time bucketization, ffill, resample, sessionization, string, datetime and binning are operation groups too, but keep `reference_implementation_class` pointing at their PyArrow Implementation; the default `reference_calculate` runs it through `calculate_with`, and the tests run the implementation under test with `calculate_with` as well. EMA has no PyArrow Implementation and pins literals instead of a live reference.

`self.test_data` is produced by the framework mixin's `create_test_data(TypedTestData)`. The input is typed data with a declared `DataType` per column; core's adapter converts it per framework, once per test. The reference side takes the same data as an Arrow table via `arrow_table_of`. This keeps the fixtures aligned: both sides compute from the *same* starting values, just in different native representations. Building PythonDict data needs no pyarrow; the other adapters, the reference comparison and the kit package itself still do.

---

## What "matching" means

For integer operations, matching means exact element-wise equality including NULLs. For floating-point operations, the test harness accepts `pytest.approx` with `rel=1e-6` when `use_approx=True`. That tolerance exists because frameworks differ in accumulation order (running sums through SQL engines vs. columnar reductions), not because results are allowed to drift.

If your framework produces NaN where the reference produces NULL, the comparison fails. NaN and NULL are not interchangeable; frame-level tests treat them as distinct.

---

## Canonical test data

Row-preserving and aggregation tests share a canonical input: 12 rows defined in `mloda/testing/data_creator/base.py` (`region`, `category`, `timestamp`, `value_int`, `value_float`, `name` and more). Each operation's test base asserts specific expected values against this fixture. When you subclass the test base for a new framework, you do not redefine the data or the expected values. The data is typed (`TypedTestData` with a declared `DataType` per column), and core's adapter converts it to your framework's native type. A fixture overrides `source_data()`.

---

## When the reference itself is wrong

Bugs happen. If the reference implementation is wrong:

1. Fix the reference implementation (PyArrow, or the pure-Python reference in `mloda/testing`).
2. Update the operation's test base (`BinningTestBase`, `AggregationTestBase`, etc.) so the expected values reflect the correct result.
3. Run every framework's test class. Every non-PyArrow implementation that was "matching" the bug will now fail.
4. Fix each framework implementation so it matches the corrected reference.

Do not "fix" a non-PyArrow framework to agree with the PyArrow bug. The reference is where correctness lives, so fixing the reference is what realigns the ecosystem.

---

## Pure Python reference for the aggregating operations

Aggregation, window aggregation, frame aggregate and scalar aggregate share one Python kernel, `mloda/testing/feature_groups/data_operations/aggregation_helpers.py`. A reference reads columns with `to_pylist()`, groups with `group_rows`, aggregates with `aggregate` and builds the output with the declared dtype. `pa.Table` is only the I/O container. Other operations do not have this reference yet (point and scalar arithmetic have their own, see [Pinned cases](#pinned-cases)). The contract, per agg type over one group's values:

| Rule | Contract |
|---|---|
| None | Skipped by every agg type. An empty or all-None group gives 0 for count and nunique, None for the rest. |
| NaN | A value, distinct from None. sum, avg, mean, std*, var* propagate it; count counts it; min and max skip it (all-NaN returns NaN); median skips it (all-NaN returns None); mode and nunique count it as one value; first and last skip only None. |
| Signed zero | `0.0` and `-0.0` are one value (nunique, mode) and one partition key. |
| std, var | Population for `std`, `var`, `std_pop`, `var_pop`; `ddof=1` for `*_samp`. Fewer than `ddof+1` values gives None. |
| mode | Ties keep the first occurrence. |
| Partition keys | NaN keys form one group, None is its own key, groups come in first-occurrence order. `partition_by=[]` is one whole-table group, also on an empty table. |
| Dtype | count and nunique int64; avg, mean, median, std*, var* float64; sum widens signed integers to int64, unsigned integers and bool to uint64, floats to float64 (a null-typed source gives float64, any other type raises `ValueError`); avg, mean, median, std* and var* accept only integer, floating or null sources and raise `ValueError` naming the type otherwise; an int64 sum that overflows raises; min, max, first, last and mode keep the source type. |

- **Order**: within a group, `order_by` (window first and last, frame) sorts ascending: real values, then NaN, then None, stable.
- **Row order**: window, frame and scalar keep input order; aggregation returns one row per group and is compared by key.
- **Tolerance**: floats compare with `pytest.approx(rel=1e-6)` where a test sets `use_approx=True`.
- **Masks**: non-matching rows become None before the aggregate. The mask semantics belong to mloda core's mask engine, so the references call the kit helper `masked_column` (`mloda/testing/feature_groups/data_operations/helpers.py`), which applies core's PyArrow mask engine, and only `to_pylist()` its output.

Declared dtypes bind the reference only; backends are compared by value. For these four operations PyArrow is not the reference. Its `0.0`/`-0.0` split is an accepted divergence (see [Known divergences](known-divergences.md)).

---

## When PyArrow has no kernel, or is the outlier

PyArrow has no exact grouped kernel for rank, offset or percentile, so a pure-Python reference in `mloda/testing` computes them. Its answer follows these rules, in order:

1. **The ungrouped PyArrow kernel.** Per group, the reference returns what PyArrow's ungrouped kernel for the same op returns on that group's values. `reference_percentile` computes it directly; the aggregating references follow the contract above, which was set from the same kernels.
2. **The majority.** Where no PyArrow kernel covers the op or the case, the majority of the frameworks that implement the op natively decides.
3. **Mode ties, the one exception.** The aggregating references break mode ties by first occurrence, while `pc.mode` picks the smallest value. Every backend already implements first occurrence, so switching would change all of them without improving agreement.

[Known divergences](known-divergences.md) records each decision. The reference and PythonDict stay separate implementations of the same rules: one shared implementation could not verify itself.

**PyArrow as the outlier.** For the operations that keep a PyArrow-based reference, PyArrow stays the reference when its convention is the odd one out: re-voting the reference per edge case brings back the drift this pattern exists to prevent. A backend that would need a hot-path branch to match such an edge case, which the canonical fixture does not exercise, keeps its native convention as an accepted divergence, with a regression test that pins its choice.

---

## Pinned cases

A reference ships `pinning_rows.py`: hand-computed rows (typed input columns, feature, context, typed expected columns) grouped under the rules they pin. The runner `mloda/community/feature_groups/data_operations/tests/test_reference_pinning.py` runs every row of its `PINNED` list against the reference with strict comparison (exact values, types and float sign) and checks that the block below equals `render_pinning_block` (`mloda/testing/feature_groups/data_operations/pinning.py`). To regenerate it, splice the rendered output between the markers. Declared types bind the reference only: for example the reference widens every integer to int64, which PyArrow does not.

<!-- BEGIN GENERATED: reference-pinning -->

Rendered from each reference's `pinning_rows.py`; every row is run against its reference.

### Rules (point_arithmetic, scalar_arithmetic)

| Case | Rule | Pinned in |
|---|---|---|
| none | None in either operand gives None, also when the other operand is NaN. | point_arithmetic, scalar_arithmetic |
| nan | NaN is a value: it propagates through add, subtract, multiply and divide. | point_arithmetic, scalar_arithmetic |
| divide_by_zero | Divide follows IEEE 754: x / 0 is inf with sign(a) * sign(b), 0 / 0 and NaN / 0 are NaN; the scalar group rejects a zero constant before compute. | point_arithmetic |
| signed_zero | A zero result keeps its IEEE sign, so 0.0 * -1.0 is -0.0. | point_arithmetic, scalar_arithmetic |
| dtype | Integer operands of any width, signed or unsigned, give int64; float64 when either operand is a float or the op is divide; an int64 overflow raises; bool, string, decimal and null-typed sources are rejected. | point_arithmetic, scalar_arithmetic |
| row_order | Output rows keep the input order, also for an empty input. | point_arithmetic, scalar_arithmetic |

### point_arithmetic

| Case | Feature | Context | Input | Expected |
|---|---|---|---|---|
| none | `a&b__add_point` | - | a=[1, None, 3] (INT64); b=[None, 2, 4] (INT64) | a&b__add_point=[None, None, 7] (INT64) |
| none | `a&b__divide_point` | - | a=[None, 1.0] (DOUBLE); b=[2.0, None] (DOUBLE) | a&b__divide_point=[None, None] (DOUBLE) |
| nan | `a&b__multiply_point` | - | a=[NaN, 2.0, NaN] (DOUBLE); b=[1.0, NaN, None] (DOUBLE) | a&b__multiply_point=[NaN, NaN, None] (DOUBLE) |
| divide_by_zero | `a&b__divide_point` | - | a=[1.0, -1.0, 1.0, -1.0, 0.0, NaN, 0.0] (DOUBLE); b=[0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0] (DOUBLE) | a&b__divide_point=[inf, -inf, -inf, inf, NaN, NaN, NaN] (DOUBLE) |
| divide_by_zero | `a&b__divide_point` | - | a=[1, -1, 0] (INT64); b=[0, 0, 0] (INT64) | a&b__divide_point=[inf, -inf, NaN] (DOUBLE) |
| signed_zero | `a&b__multiply_point` | - | a=[0.0, -0.0, -0.0] (DOUBLE); b=[-1.0, -1.0, 1.0] (DOUBLE) | a&b__multiply_point=[-0.0, 0.0, -0.0] (DOUBLE) |
| signed_zero | `a&b__add_point` | - | a=[-0.0, -0.0, 0.0] (DOUBLE); b=[-0.0, 0.0, -0.0] (DOUBLE) | a&b__add_point=[-0.0, 0.0, 0.0] (DOUBLE) |
| signed_zero | `a&b__subtract_point` | - | a=[0.0, -0.0, 0.0] (DOUBLE); b=[0.0, 0.0, -0.0] (DOUBLE) | a&b__subtract_point=[0.0, -0.0, 0.0] (DOUBLE) |
| signed_zero | `a&b__divide_point` | - | a=[0.0, -0.0] (DOUBLE); b=[-1.0, -1.0] (DOUBLE) | a&b__divide_point=[-0.0, 0.0] (DOUBLE) |
| dtype | `a&b__add_point` | - | a=[2147483647] (INT32); b=[1] (INT32) | a&b__add_point=[2147483648] (INT64) |
| dtype | `a&b__multiply_point` | - | a=[100, -100] (INT8); b=[100, 100] (INT8) | a&b__multiply_point=[10000, -10000] (INT64) |
| dtype | `a&b__subtract_point` | - | a=[1] (UINT8); b=[2] (UINT8) | a&b__subtract_point=[-1] (INT64) |
| dtype | `a&b__add_point` | - | a=[5] (UINT64); b=[7] (UINT64) | a&b__add_point=[12] (INT64) |
| dtype | `a&b__add_point` | - | a=[1, 2] (INT64); b=[0.5, 0.25] (DOUBLE) | a&b__add_point=[1.5, 2.25] (DOUBLE) |
| dtype | `a&b__add_point` | - | a=[1] (INT32); b=[0.5] (FLOAT) | a&b__add_point=[1.5] (DOUBLE) |
| dtype | `a&b__divide_point` | - | a=[7, -7] (INT64); b=[2, 2] (INT64) | a&b__divide_point=[3.5, -3.5] (DOUBLE) |
| row_order | `a&b__add_point` | - | a=[3, 1, 2] (INT64); b=[30, 10, 20] (INT64) | a&b__add_point=[33, 11, 22] (INT64) |
| row_order | `a&b__add_point` | - | a=[] (INT64); b=[] (INT64) | a&b__add_point=[] (INT64) |

### scalar_arithmetic

| Case | Feature | Context | Input | Expected |
|---|---|---|---|---|
| none | `a__add_constant` | constant=1 | a=[1, None, 3] (INT64) | a__add_constant=[2, None, 4] (INT64) |
| nan | `a__multiply_constant` | constant=2.0 | a=[NaN, None, 1.5] (DOUBLE) | a__multiply_constant=[NaN, None, 3.0] (DOUBLE) |
| signed_zero | `a__multiply_constant` | constant=-0.0 | a=[1.0, -1.0, 0.0, -0.0] (DOUBLE) | a__multiply_constant=[-0.0, 0.0, -0.0, 0.0] (DOUBLE) |
| signed_zero | `a__multiply_constant` | constant=-1 | a=[0.0, -0.0, 2.0] (DOUBLE) | a__multiply_constant=[-0.0, 0.0, -2.0] (DOUBLE) |
| dtype | `a__add_constant` | constant=0.5 | a=[1, 2] (INT64) | a__add_constant=[1.5, 2.5] (DOUBLE) |
| dtype | `a__add_constant` | constant=1 | a=[2147483647] (INT32) | a__add_constant=[2147483648] (INT64) |
| dtype | `a__multiply_constant` | constant=3 | a=[1.0, 2.0] (DOUBLE) | a__multiply_constant=[3.0, 6.0] (DOUBLE) |
| dtype | `a__divide_constant` | constant=2 | a=[7, -7] (INT64) | a__divide_constant=[3.5, -3.5] (DOUBLE) |
| dtype | `a__subtract_constant` | constant=2 | a=[1] (UINT8) | a__subtract_constant=[-1] (INT64) |
| row_order | `a__subtract_constant` | constant=1 | a=[3, 1, 2] (INT64) | a__subtract_constant=[2, 0, 1] (INT64) |
| row_order | `a__multiply_constant` | constant=2 | a=[] (INT64) | a__multiply_constant=[] (INT64) |

<!-- END GENERATED: reference-pinning -->

---

## Related

- [Row-preserving contract](02-row-preserving-contract.md) - The row-count-and-order invariant that the reference comparison enforces.
- [Supported ops per framework](04-supported-ops.md) - How a framework skips comparisons for ops it cannot express.
- [Known divergences](known-divergences.md) - Audited cases where a framework's native operator would diverge from the reference, and the fix or exclusion used.
- [Adding a new data operation](10-adding-new-operation.md) - Where to put the PyArrow reference when adding an op.
