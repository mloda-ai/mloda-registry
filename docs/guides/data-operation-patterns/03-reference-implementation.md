# Reference Implementation Pattern

Every data operation has one framework designated as the reference. Other frameworks are correct if and only if they match the reference on the shared test suite. The aggregating operations (aggregation, window aggregation, frame aggregate, scalar aggregate) have a pure Python reference. PyArrow is the reference for most other operations that ship today; where it has no exact grouped kernel, a pure-Python reference applies PyArrow's semantics (see [When PyArrow has no kernel](#when-pyarrow-has-no-kernel-or-is-the-outlier)).

**What**: One framework implementation is the source of truth. Cross-framework tests run the same feature through both the target framework and the reference, then assert the two results match.
**When**: Every time you add a new operation or add a new framework implementation for an existing operation.
**Why**: Without a fixed reference, frameworks drift into mutually incompatible results (e.g. different NULL handling, different tie-breaking in ranks). For the operations that keep a PyArrow-based reference, making PyArrow authoritative keeps semantics aligned without requiring everyone to agree on an abstract spec; the aggregating operations pin the same idea in a written contract instead.
**Where**: `mloda/testing/feature_groups/data_operations/base.py::DataOpsTestBase._compare_with_reference`.
**How**: Each framework mixin provides `create_test_data` (converts the canonical PyArrow table to the framework's native format) and `extract_column` (materializes the result back to a Python list for comparison).

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

Aggregation, window aggregation, frame aggregate, scalar aggregate, rank, offset and percentile are the exceptions: each is one `OperationFeatureGroup` served by one `Implementation` per framework, so their test bases have no `reference_implementation_class`. Their references are the functions `reference_aggregation`, `reference_window_aggregation`, `reference_frame_aggregate`, `reference_scalar_aggregate`, `reference_rank`, `reference_offset` and `reference_percentile` (each `(group, table, features)`, in the operation's `reference.py` under `mloda/testing/feature_groups/data_operations/`), which run the parsing loop of the group under test with the kit's compute kernel and import nothing from the code they judge; `reference_calculate` passes `feature_group_of(implementation_class())` and reaches them, and the test kit runs the implementation with `calculate_with` (`mloda/testing/feature_groups/data_operations/helpers.py`). Their NaN divergences come from the implementation's shipped `divergences` records. Point and scalar arithmetic, lookup, time bucketization, ffill, resample, sessionization, string, datetime and binning are operation groups too, but keep `reference_implementation_class` pointing at their PyArrow Implementation; the default `reference_calculate` runs it through `calculate_with`, and the tests run the implementation under test with `calculate_with` as well. EMA has no PyArrow Implementation and pins literals instead of a live reference.

`self.test_data` is produced by the framework mixin's `create_test_data(pa.Table)`. The canonical input is always PyArrow; the mixin converts it once per test. This keeps the fixtures aligned: both sides compute from the *same* starting values, just in different native representations.

---

## What "matching" means

For integer operations, matching means exact element-wise equality including NULLs. For floating-point operations, the test harness accepts `pytest.approx` with `rel=1e-6` when `use_approx=True`. That tolerance exists because frameworks differ in accumulation order (running sums through SQL engines vs. columnar reductions), not because results are allowed to drift.

If your framework produces NaN where the reference produces NULL, the comparison fails. NaN and NULL are not interchangeable; frame-level tests treat them as distinct.

---

## Canonical test data

Row-preserving and aggregation tests share a canonical input: 12 rows defined in `mloda/testing/data_creator/base.py` (`region`, `category`, `timestamp`, `value_int`, `value_float`, `name` and more). Each operation's test base asserts specific expected values against this fixture. When you subclass the test base for a new framework, you do not redefine the data or the expected values. You only implement the adapter methods that convert between PyArrow and your framework's native type.

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

Aggregation, window aggregation, frame aggregate and scalar aggregate share one Python kernel, `mloda/testing/feature_groups/data_operations/aggregation_helpers.py`. A reference reads columns with `to_pylist()`, groups with `group_rows`, aggregates with `aggregate` and builds the output with the declared dtype. `pa.Table` is only the I/O container. Other operations do not have this reference yet. The rules (None, NaN, signed zero, std and var, mode, partition keys, dtype, order, row order) and their pinned rows are in [Pinned cases](#pinned-cases). Aggregation returns one row per group and is compared by key. avg, mean, median, std* and var* accept only integer, floating or null sources and raise `ValueError` naming the type otherwise; sum over a null-typed source gives float64, any other non-numeric type raises `ValueError`; an int64 sum that overflows raises.

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

Each reference ships `pinning_rows.py` next to it: hand-computed rows, never produced by running the reference. `mloda/community/feature_groups/data_operations/tests/test_reference_pinning.py` runs every row against its reference and checks this block; regenerate it from `render_pinning_block` when a row changes.

<!-- BEGIN GENERATED: reference-pinning -->

Rendered from each reference's `pinning_rows.py`; every row is run against its reference.

### Rules (aggregation, frame_aggregate, scalar_aggregate, window_aggregation)

| Case | Rule | Pinned in |
|---|---|---|
| none | Skipped by every agg type; an empty or all-None group gives 0 for count and nunique, None for the rest. | aggregation, frame_aggregate, scalar_aggregate, window_aggregation |
| nan | A value, distinct from None: sum, avg, mean, std* and var* propagate it; count counts it; min and max skip it (all-NaN returns NaN); median skips it (all-NaN returns None); mode and nunique count it as one value; first and last skip only None. | aggregation, frame_aggregate, scalar_aggregate, window_aggregation |
| signed_zero | 0.0 and -0.0 are one value (nunique, mode) and one partition key. | aggregation, frame_aggregate |
| std_var | Population for std, var, std_pop and var_pop; ddof=1 for the *_samp types; fewer than ddof+1 values gives None. | aggregation, frame_aggregate, scalar_aggregate |
| mode | Ties keep the first occurrence. | aggregation |
| partition_keys | NaN keys form one group, None is its own key, groups come in first-occurrence order; partition_by=[] is one whole-table group, also on an empty table. | aggregation, frame_aggregate |
| dtype | count and nunique are int64; avg, mean, median, std* and var* are float64; sum widens signed integers to int64, unsigned integers to uint64 and floats to float64; min, max, first, last and mode keep the source type. | aggregation, frame_aggregate, scalar_aggregate, window_aggregation |
| order | Within a group, order_by sorts ascending: real values, then NaN, then None, stable. | frame_aggregate, window_aggregation |
| row_order | Window, frame and scalar keep input order. | frame_aggregate, scalar_aggregate, window_aggregation |

### aggregation

| Case | Feature | Context | Input | Expected |
|---|---|---|---|---|
| none | `v__sum_agg` | partition_by=[g] | g=[a, a, b]; v=[1, None, 5] | g=[a, b]; v__sum_agg=[1, 5] |
| none | `v__count_agg` | partition_by=[g] | g=[a, a, a]; v=[1, None, 3] | v__count_agg=[2] |
| none | `v__count_agg` | partition_by=[g] | g=[a, a, b, b]; v=[None, None, 2, 3] | v__count_agg=[0, 2] |
| none | `v__sum_agg` | partition_by=[g] | g=[a, a, b, b]; v=[None, None, 2, 3] | v__sum_agg=[None, 5] |
| nan | `v__sum_agg` | partition_by=[g] | g=[a, a, b]; v=[1.0, NaN, 2.0] | v__sum_agg=[NaN, 2.0] |
| nan | `v__count_agg` | partition_by=[g] | g=[a, a]; v=[NaN, 1.0] | v__count_agg=[2] |
| nan | `v__min_agg` | partition_by=[g] | g=[a, a, b]; v=[NaN, 1.0, NaN] | v__min_agg=[1.0, NaN] |
| nan | `v__max_agg` | partition_by=[g] | g=[a, a, b]; v=[NaN, 1.0, NaN] | v__max_agg=[1.0, NaN] |
| nan | `v__median_agg` | partition_by=[g] | g=[a, a, a, b]; v=[NaN, 1.0, 3.0, NaN] | v__median_agg=[2.0, None] |
| nan | `v__first_agg` | partition_by=[g] | g=[a, a, a]; v=[None, NaN, 1.0] | v__first_agg=[NaN] |
| nan | `v__nunique_agg` | partition_by=[g] | g=[a, a, a, a]; v=[NaN, NaN, 1.0, None] | v__nunique_agg=[2] |
| signed_zero | `v__nunique_agg` | partition_by=[g] | g=[a, a, a]; v=[0.0, -0.0, -0.0] | v__nunique_agg=[1] |
| signed_zero | `v__sum_agg` | partition_by=[g] | g=[0.0, -0.0]; v=[1, 2] | g=[0.0]; v__sum_agg=[3] |
| mode | `v__mode_agg` | partition_by=[g] | g=[a, a, a, a, a]; v=[2, 1, 1, 2, 3] | v__mode_agg=[2] (int64) |
| std_var | `v__std_pop_agg` | partition_by=[g] | g=[a, a, b]; v=[1.0, 3.0, 5.0] | v__std_pop_agg=[1.0, 0.0] |
| std_var | `v__var_samp_agg` | partition_by=[g] | g=[a, a, b]; v=[1.0, 3.0, 5.0] | v__var_samp_agg=[2.0, None] |
| std_var | `v__std_samp_agg` | partition_by=[g] | g=[a, a, b]; v=[1.0, 3.0, 5.0] | v__std_samp_agg=[1.4142135623730951, None] |
| partition_keys | `v__sum_agg` | partition_by=[g] | g=[NaN, None, 1.0, NaN, None]; v=[1, 2, 3, 4, 5] | g=[NaN, None, 1.0]; v__sum_agg=[5, 7, 3] |
| partition_keys | `v__count_agg` | partition_by=[] | v=[] (int64) | v__count_agg=[0] |
| dtype | `v__sum_agg` | partition_by=[g] | g=[a, a]; v=[1, 2] (int32) | v__sum_agg=[3] (int64) |
| dtype | `v__sum_agg` | partition_by=[g] | g=[a, a]; v=[200, 100] (uint8) | v__sum_agg=[300] (uint64) |
| dtype | `v__count_agg` | partition_by=[g] | g=[a]; v=[1] (int32) | v__count_agg=[1] (int64) |
| dtype | `v__avg_agg` | partition_by=[g] | g=[a, a]; v=[1, 2] (int32) | v__avg_agg=[1.5] (float64) |
| dtype | `v__min_agg` | partition_by=[g] | g=[a, a]; v=[4, 2] (int32) | v__min_agg=[2] (int32) |

### frame_aggregate

| Case | Feature | Context | Input | Expected |
|---|---|---|---|---|
| none | `value__sum_rolling_3` | partition_by=[region], order_by=ts | region=[A, A, A, A]; ts=[1, 2, 3, 4]; value=[10, None, 30, 40] | value__sum_rolling_3=[10, 10, 40, 70] |
| none | `value__cumsum` | partition_by=[region], order_by=ts | region=[A, A]; ts=[1, 2]; value=[None, None] | value__cumsum=[None, None] |
| none | `value__cumcount` | partition_by=[region], order_by=ts | region=[A, A]; ts=[1, 2]; value=[None, None] | value__cumcount=[0, 0] |
| none | `value__sum_rolling_2` | partition_by=[region], order_by=ts | region=[A, A, A]; ts=[1, 2, 3]; value=[None, None, 5] | value__sum_rolling_2=[None, None, 5] |
| nan | `value__avg_rolling_2` | partition_by=[region], order_by=ts | region=[A, A, A, A]; ts=[1, 2, 3, 4]; value=[1.0, NaN, 3.0, 5.0] | value__avg_rolling_2=[1.0, NaN, NaN, 4.0] |
| nan | `value__max_rolling_2` | partition_by=[region], order_by=ts | region=[A, A, A, A]; ts=[1, 2, 3, 4]; value=[1.0, NaN, 3.0, 5.0] | value__max_rolling_2=[1.0, 1.0, 3.0, 5.0] |
| std_var | `value__std_rolling_2` | partition_by=[region], order_by=ts | region=[A, A, A]; ts=[1, 2, 3]; value=[1.0, 3.0, 3.0] | value__std_rolling_2=[0.0, 1.0, 0.0] |
| std_var | `value__var_rolling_2` | partition_by=[region], order_by=ts | region=[A, A, A]; ts=[1, 2, 3]; value=[1.0, 3.0, 3.0] | value__var_rolling_2=[0.0, 1.0, 0.0] |
| order | `value__cumsum` | partition_by=[region], order_by=ts | region=[A, A, A]; ts=[None, 1, 2]; value=[100, 10, 20] | value__cumsum=[130, 10, 30] |
| order | `value__cumsum` | partition_by=[region], order_by=ts | region=[A, A, A, A]; ts=[None, 3.0, NaN, 1.0]; value=[100, 4, 8, 1] | value__cumsum=[113, 5, 13, 1] |
| row_order | `value__cumsum` | partition_by=[region], order_by=ts | region=[A, A, A, A]; ts=[3.0, 1.0, NaN, 2.0]; value=[4, 1, 8, 2] | value__cumsum=[7, 1, 15, 3] |
| partition_keys | `value__cumsum` | partition_by=[region], order_by=ts | region=[NaN, NaN, None, 1.0, None]; ts=[1, 2, 3, 4, 5]; value=[1, 2, 3, 4, 5] | value__cumsum=[1, 3, 3, 4, 8] |
| signed_zero | `value__cumsum` | partition_by=[region], order_by=ts | region=[0.0, -0.0, 0.0]; ts=[1, 2, 3]; value=[1, 2, 4] | value__cumsum=[1, 3, 7] |
| dtype | `value__cumsum` | partition_by=[region], order_by=ts | region=[A, A]; ts=[1, 2]; value=[1, 2] (int32) | value__cumsum=[1, 3] (int64) |
| dtype | `value__cumavg` | partition_by=[region], order_by=ts | region=[A, A]; ts=[1, 2]; value=[1, 2] (int32) | value__cumavg=[1.0, 1.5] (float64) |

### scalar_aggregate

| Case | Feature | Context | Input | Expected |
|---|---|---|---|---|
| none | `v__sum_scalar` | - | v=[1, None, 5] | v__sum_scalar=[6, 6, 6] |
| none | `v__count_scalar` | - | v=[None, None] (int64) | v__count_scalar=[0, 0] |
| none | `v__std_scalar` | - | v=[None, None] (int64) | v__std_scalar=[None, None] |
| nan | `v__avg_scalar` | - | v=[1.0, NaN, 3.0] | v__avg_scalar=[NaN, NaN, NaN] |
| nan | `v__median_scalar` | - | v=[NaN, 1.0, 3.0] | v__median_scalar=[2.0, 2.0, 2.0] |
| nan | `v__max_scalar` | - | v=[NaN, NaN] | v__max_scalar=[NaN, NaN] |
| std_var | `v__var_samp_scalar` | - | v=[2.0, 4.0] | v__var_samp_scalar=[2.0, 2.0] |
| std_var | `v__std_pop_scalar` | - | v=[5.0] | v__std_pop_scalar=[0.0] |
| std_var | `v__var_samp_scalar` | - | v=[5.0] | v__var_samp_scalar=[None] |
| row_order | `v__min_scalar` | - | v=[3, 1, 2] | v=[3, 1, 2]; v__min_scalar=[1, 1, 1] |
| dtype | `v__sum_scalar` | - | v=[200, 100] (uint8) | v__sum_scalar=[300, 300] (uint64) |
| dtype | `v__avg_scalar` | - | v=[1, 2] (int32) | v__avg_scalar=[1.5, 1.5] (float64) |

### window_aggregation

| Case | Feature | Context | Input | Expected |
|---|---|---|---|---|
| none | `v__sum_window` | partition_by=[g] | g=[a, b, a, b, a]; v=[1, None, 2, 4, None] | v__sum_window=[3, 4, 3, 4, 3] |
| none | `v__count_window` | partition_by=[g] | g=[a, a, b]; v=[None, None, 7] | v__count_window=[0, 0, 1] |
| none | `v__sum_window` | partition_by=[g] | g=[a, a, b]; v=[None, None, 7] | v__sum_window=[None, None, 7] |
| none | `v__first_window` | partition_by=[g], order_by=o | g=[a, a]; v=[None, 7]; o=[1, 2] | v__first_window=[7, 7] |
| nan | `v__avg_window` | partition_by=[g] | g=[a, a, b]; v=[1.0, NaN, 2.0] | v__avg_window=[NaN, NaN, 2.0] |
| nan | `v__median_window` | partition_by=[g] | g=[a, a, a]; v=[NaN, 1.0, 3.0] | v__median_window=[2.0, 2.0, 2.0] |
| order | `v__first_window` | partition_by=[g], order_by=o | g=[a, a, a, a]; v=[10, 20, 30, 40]; o=[3.0, NaN, None, 1.0] | v__first_window=[40, 40, 40, 40] |
| order | `v__last_window` | partition_by=[g], order_by=o | g=[a, a, a, a]; v=[10, 20, 30, 40]; o=[3.0, NaN, None, 1.0] | v__last_window=[30, 30, 30, 30] |
| order | `v__first_window` | partition_by=[g], order_by=o | g=[a, a, a]; v=[1, 2, 3]; o=[5, 5, 5] | v__first_window=[1, 1, 1] |
| order | `v__last_window` | partition_by=[g], order_by=o | g=[a, a, a]; v=[1, 2, 3]; o=[5, 5, 5] | v__last_window=[3, 3, 3] |
| row_order | `v__min_window` | partition_by=[g] | g=[b, a, b, a]; v=[1, 10, 2, 20] | v__min_window=[1, 10, 1, 10] |
| dtype | `v__sum_window` | partition_by=[g] | g=[a, a]; v=[1, 2] (int32) | v__sum_window=[3, 3] (int64) |

<!-- END GENERATED: reference-pinning -->

---

## Related

- [Row-preserving contract](02-row-preserving-contract.md) - The row-count-and-order invariant that the reference comparison enforces.
- [Supported ops per framework](04-supported-ops.md) - How a framework skips comparisons for ops it cannot express.
- [Known divergences](known-divergences.md) - Audited cases where a framework's native operator would diverge from the reference, and the fix or exclusion used.
- [Adding a new data operation](10-adding-new-operation.md) - Where to put the PyArrow reference when adding an op.
