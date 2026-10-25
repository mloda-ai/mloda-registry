# Adding a New Data Operation

End-to-end recipe for introducing a new data operation: one `OperationFeatureGroup`, one `Implementation` per framework, test base, test mixins, and how to wire everything so the cross-framework comparison works.

**What**: The concrete steps to add a new operation category (or a new op inside an existing category) and have it tested across every supported framework.
**When**: You want a declarative transform that behaves identically on PyArrow, Pandas, Polars, DuckDB, and SQLite.
**Why**: The existing pattern enforces the row-preserving contract, reference-implementation comparison, and supported-ops skipping for free. Following it gets you coverage without reinventing test harnesses.
**Where**: Code in `mloda/community/feature_groups/data_operations/{row_preserving|row_changing}/{your_category}/`, test bases in the same layout under `mloda/testing/feature_groups/data_operations/`.
**How**: Follow the seven steps below in order; every existing category was built the same way. They live under `mloda/community/feature_groups/data_operations/`, split by row-count preservation:

- `row_preserving/`: `binning`, `datetime`, `ema`, `ffill`, `frame_aggregate`, `lookup`, `offset`, `percentile`, `point_arithmetic`, `rank`, `scalar_aggregate`, `scalar_arithmetic`, `sessionization`, `time_bucketization`, `window_aggregation`, `arithmetic` (shared base, not a category)
- `row_changing/`: `resample`
- directly under `data_operations/`: `aggregation`, `string`

Regenerate this list:

```bash
git ls-tree -r HEAD --name-only mloda/community/feature_groups/data_operations/ | grep -E '/base\.py$' | grep -v '^mloda/community/feature_groups/data_operations/base\.py$'
```

---

## Step 1: Decide the category

- **Row-preserving**? Place it under `row_preserving/{your_op}/`. Read [the row-preserving contract](02-row-preserving-contract.md) first.
- **Row-reducing** (group aggregate)? Place it under `aggregation/` if it fits the `__{agg}_agg` naming; otherwise it probably wants its own folder at the top level of `data_operations/`.
- **Element-wise on strings**? Extend `string/` only if the new op has no parameters. Otherwise build a separate feature group.

The category decides your naming pattern. See [overview](01-overview.md) for the conventions.

---

## Step 2: Write the operation group

File: `mloda/community/feature_groups/data_operations/{category}/{your_op}/base.py`.

```py
from collections.abc import Callable
from typing import Any

from mloda.provider import (
    DefaultOptionKeys,
    FeatureChainParserMixin,
    FeatureSet,
    OperationFeatureGroup,
    SubtypeDeclaration,
    property_spec,
)

from mloda.community.feature_groups.data_operations.base import (
    OP_TOKEN_EXPECTED,
    assert_source_columns_present,
    is_op_token,
)

YOUR_OPS = {
    "op_a": "What op_a does",
    "op_b": "What op_b does",
}


class YourOpFeatureGroup(FeatureChainParserMixin, OperationFeatureGroup):
    PREFIX_PATTERN = r".+__(op_a|op_b)$"

    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    YOUR_OP = "your_op"

    SUBTYPES = SubtypeDeclaration(key=YOUR_OP)

    PROPERTY_MAPPING = {
        YOUR_OP: property_spec(
            "Operation applied to the source column",
            strict=True,
            allowed_values=YOUR_OPS,
            match_guard=is_op_token,
            expected=OP_TOKEN_EXPECTED,
        ),
        DefaultOptionKeys.in_features: property_spec("Source column"),
    }

    @classmethod
    def apply_your_op(
        cls,
        data: Any,
        features: FeatureSet,
        *,
        compute_your_op: Callable[[Any, str, str, str], Any],
    ) -> Any:
        """Shared loop: extract params from each feature, delegate to ``compute_your_op``."""
        table = data
        for feature in features.features:
            source_col = cls._extract_single_source_feature(feature)
            assert_source_columns_present(data, [source_col])
            op = cls._extract_op(feature)
            table = compute_your_op(table, feature.name, source_col, op)
        return table
```

The group owns:

- The base order: mixins first, `OperationFeatureGroup` last, so the mixin's matcher runs before the bare `FeatureGroup` one.
- The feature-name regex, `PROPERTY_MAPPING`, and `SUBTYPES` (the key `supported` and `divergent` are keyed by).
- The shared loop as a classmethod that takes the framework step as a keyword-only callable. It defines no step itself and no `compute_framework_rule`: the frameworks are the ones its Implementations serve.
- Extraction of source column, operation, and any options from `Options(context=...)`.
- The shared `assert_source_columns_present` presence guard, called once per feature before any per-backend dtype guard.
- The shared `assert_key_columns_present` guard for `partition_by`, `order_by` (or `time_column`) and mask columns, called once per feature before the framework step.

Existing groups to crib from: `row_preserving/binning/base.py` (simple), `row_preserving/window_aggregation/base.py` (with `partition_by`/`order_by`/masks). They compose `FeatureChainParserMixin` to parse the suffix of the feature name; copy that detail verbatim from the closest existing group. Single-input ops use `_extract_single_source_feature`, multi-input ops `_extract_validated_source_features` (see `row_preserving/point_arithmetic/base.py`), so the MIN/MAX in_features and empty-operand checks live in the core helper, not a per-op override. Likewise, inherit `input_features`, `_extract_source_features` (a single-input op needs no override of it), and the match-time MIN/MAX check from the mixin instead of overriding them.

---

## Step 3: Write the PyArrow Implementation first

PyArrow is the reference. Write it first; it defines correctness for everything else.

File: `{category}/{your_op}/pyarrow_{your_op}.py`.

```py
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.{category}.{your_op}.base import YourOpFeatureGroup


class PyArrowYourOp(Implementation[YourOpFeatureGroup]):
    feature_group = YourOpFeatureGroup
    compute_framework = PyArrowTable

    @classmethod
    def calculate(
        cls, group: type[YourOpFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.apply_your_op(data, features, compute_your_op=cls._compute_your_op)

    @classmethod
    def _compute_your_op(cls, data: pa.Table, feature_name: str, source_col: str, op: str) -> pa.Table:
        col = data.column(source_col)
        # ...op-specific columnar work...
        return data.append_column(feature_name, result_array)
```

Preserve row order. Preserve NULL. Handle the empty-table case.

Every family has this shape. The reference of aggregation, window, frame and scalar aggregate, rank, offset and percentile is a function (`reference_aggregation`, `reference_window_aggregation`, `reference_scalar_aggregate`, `reference_rank`, ...), the others' is their PyArrow Implementation (EMA pins literals). Rank and offset also declare `SUBTYPES` with parametric families (`ntile_N`, `top_N`, `bottom_N`; `lag_N`, `lead_N`, ...), which core collapses to the family name; lookup declares no `SUBTYPES`, so its Implementations declare neither `supported` nor `divergent`. See [Framework-Specific](../feature-group-patterns/09-framework-specific.md).

---

## Step 4: Write the operation's test bases

File: `mloda/testing/feature_groups/data_operations/{category}/{your_op}/{your_op}.py`.

```py
from typing import Any
import pytest
from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase


class YourOpTestBase(DataOpsTestBase):
    @classmethod
    def reference_implementation_class(cls) -> Any:
        from mloda.community.feature_groups.data_operations.{category}.{your_op}.pyarrow_{your_op} import (
            PyArrowYourOp,
        )

        return PyArrowYourOp

    @classmethod
    def supported_ops(cls) -> set[str]:
        return {"op_a", "op_b"}

    @pytest.mark.parametrize(
        ("op", "partition_by"),
        [
            pytest.param("op_a", None, id="op_a"),
            pytest.param("op_b", ["region"], id="op_b"),
        ],
    )
    def test_cross_framework(self, op: str, partition_by: list[str] | None) -> None:
        self._skip_if_unsupported(op)
        self._compare_with_reference(f"value_int__{op}", partition_by=partition_by)
```

The test method calls `_compare_with_reference`, which runs both the Implementation under test and the PyArrow reference and asserts they match. Every framework's concrete test class inherits it.

One parametrized method per family, not one method per op: the explicit ids name the failing case (`test_cross_framework[op_b]`), and calling `_skip_if_unsupported` inside the body keeps the skip per case, so an unsupported op skips alone.

The group's shape gets its own test in `{category}/{your_op}/tests/test_base.py`: subclass `OperationGroupTestBase` (`mloda/testing/feature_groups/data_operations/operation_group.py`) and fill its hooks. It checks the base order, `SUBTYPES`, that the loop is on the group and the step only on the Implementations, each Implementation's framework pair, `supported` and `divergent`, and the manifest:

```py
from mloda.testing.feature_groups.data_operations.operation_group import ImplSpec, OperationGroupTestBase

_ENGINE = "mloda_plugins.compute_framework.base_implementations"


class TestYourOpOperationGroup(OperationGroupTestBase):
    @classmethod
    def group_class(cls) -> Any:
        return YourOpFeatureGroup

    @classmethod
    def backend_package(cls) -> str:
        return "mloda.community.feature_groups.data_operations.{category}.{your_op}"

    @classmethod
    def implementations(cls) -> list[ImplSpec]:
        return [
            ImplSpec("pyarrow", "pyarrow_{your_op}", "PyArrowYourOp", f"{_ENGINE}.pyarrow.table", "PyArrowTable"),
            ImplSpec(
                "sqlite",
                "sqlite_{your_op}",
                "SqliteYourOp",
                f"{_ENGINE}.sqlite.sqlite_framework",
                "SqliteFramework",
                supported=frozenset({"op_a"}),
                divergent=frozenset({"op_a"}),
            ),
            # ...one ImplSpec per framework; dependency="pandas" etc. for optional backends
        ]

    @classmethod
    def subtype_key(cls) -> str:
        return "your_op"

    @classmethod
    def loop_method(cls) -> str:
        return "apply_your_op"

    @classmethod
    def compute_method(cls) -> str:
        return "_compute_your_op"

    # Plus probe_feature_set, name_probes, option_probes (see binning/tests/test_base.py)
```

---

## Step 5: Add the other framework Implementations

One file per framework, each an `Implementation[YourOpFeatureGroup]`, plus the manifest:

```text
{category}/{your_op}/
  base.py
  manifest.py
  pyarrow_{your_op}.py
  pandas_{your_op}.py
  polars_lazy_{your_op}.py
  duckdb_{your_op}.py
  sqlite_{your_op}.py
  python_dict_{your_op}.py
```

Each sets `feature_group` and `compute_framework`, makes `calculate` the one-liner through `group.apply_your_op`, and implements `_compute_your_op` with its native primitives. For row-preserving ops, respect the row-order invariant. If the native operation reorders (DuckDB `NTILE`), use the `ROW_NUMBER()` tag-and-restore pattern from [the row-preserving contract](02-row-preserving-contract.md).

A framework that cannot run an op, or differs from the reference on one, says so on the Implementation:

```py
class SqliteYourOp(Implementation[YourOpFeatureGroup]):
    feature_group = YourOpFeatureGroup
    compute_framework = SqliteFramework
    supported = frozenset({"op_a"})  # op_b: this framework is not offered
    divergent = {"op_a": "returns NULL on divide by zero"}
```

Reject an op the framework cannot run natively through `supported`; do not compute it in Python inside the Implementation. Examples: `aggregation/sqlite_aggregation.py` (both), `point_arithmetic/sqlite_point_arithmetic.py` (`divergent` only).

`manifest.py` lists the group and every Implementation through `load_plugin_classes`, so a backend whose optional framework is missing is skipped. Copy `row_preserving/binning/manifest.py`.

---

## Step 6: Wire up the framework test classes

One file per framework, all inheriting from your test base and the framework's mixin:

```py
# tests/test_pandas.py
from typing import Any
import pytest

pytest.importorskip("pandas")

from mloda.community.feature_groups.data_operations.{category}.{your_op}.pandas_{your_op} import (
    PandasYourOp,
)
from mloda.testing.feature_groups.data_operations.mixins.pandas import PandasTestMixin
from mloda.testing.feature_groups.data_operations.{category}.{your_op}.{your_op} import (
    YourOpTestBase,
)


class TestPandasYourOp(PandasTestMixin, YourOpTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PandasYourOp
```

No test method bodies. Everything is inherited. `implementation_class` returns the Implementation; the helpers run it through its group. If the Implementation declares `supported`, override `supported_ops` to return the same subset; `_skip_if_unsupported` handles the skip. To also pin the match-time capability per framework, mix in `CapabilityHookTestMixin` (see `aggregation/tests/test_sqlite.py`).

`PandasTestMixin` stores an int column with nulls as float64. To also run the suite on pandas nullable ints (`Int64` and friends), add a second class that swaps in `PandasNullableIntTestMixin`, e.g. `class TestPandasYourOpNullableInt(PandasNullableIntTestMixin, TestPandasYourOp)`, and reset any flags the float64 class sets for lossy ints.

---

## Step 7: Decide if it needs mask support

If the new op is an aggregate (or anything that consumes source values that a user might want to conditionally include), wire it up to `MaskTestMixin`:

```py
from mloda.testing.feature_groups.data_operations.mixins.mask import MaskTestMixin


class YourOpTestBase(MaskTestMixin, DataOpsTestBase):
    @classmethod
    def mask_feature_name(cls) -> str:
        return "value_int__op_a"

    @classmethod
    def mask_partition_by(cls) -> list[str] | None:
        return ["region"]

    @classmethod
    def mask_is_reducing(cls) -> bool:
        return False  # row-preserving

    @classmethod
    def mask_expected_row_count(cls) -> int:
        return 12

    # Plus mask_equal_expected, mask_multiple_conditions_expected,
    # mask_is_in_expected, mask_greater_than_expected, mask_no_mask_expected
```

`MaskTestMixin` adds inherited test methods covering a missing mask column, equal, AND-combined, `is_in`, greater-than, fully-masked, no-mask-baseline, and missing-value (null and NaN) scenarios. See [Masking](../feature-group-patterns/25-masking.md) for the full user-facing spec.

If the new op has a documented NaN policy (see [the reference-implementation guide](03-reference-implementation.md#when-pyarrow-has-no-kernel-or-is-the-outlier)), wire it up to `NanPolicyTestMixin` (`mloda/testing/feature_groups/data_operations/mixins/nan_policy.py`) instead of writing a bespoke `test_nan_policy_*` method. Override `nan_policy_cases` (case id -> expected value, required) and `nan_policy_feature_name` (case id -> feature name, required); `nan_policy_agg_type` defaults to the case id, `nan_policy_feature_set` defaults to partitioning by `"grp"`, `nan_policy_divergent_cases` defaults to no divergences, and `nan_policy_extract_values` defaults to a plain list column. A divergent case only applies for a concrete framework test class that names its agg type (via `nan_policy_agg_type`) in that class's `nan_divergent_agg_types()`, which by default is the Implementation's shipped `divergent` keys (empty for a plain FeatureGroup subclass); a framework with no divergence for the op does not need to override `nan_policy_divergent_cases` at all. The default skip hook calls `_skip_if_unsupported`, which needs a `supported_*` set on the base (e.g. `supported_agg_types`); ops without one (resample, percentile) must override `nan_policy_skip_if_unsupported` instead.

A new case-parametrized test mixin inherits `CaseParametrizationTestMixin` (`mloda/testing/feature_groups/data_operations/mixins/case_parametrization.py`) and declares its fixture in `_case_fixtures`, rather than defining its own `pytest_generate_tests`; `mixins/tests/test_mixin_isolation.py` enforces this structurally.

Every test base also mixes in `OutputContractTestMixin` (`mloda/testing/feature_groups/data_operations/mixins/output_contract.py`), which covers the result-type, row-count, and new-column checks that used to be hand-written per base. Override `output_contract_feature_set` (an instance method) to return the `FeatureSet` to exercise, and `output_contract_expected_row_count` when the op changes the row count (return the expected count for the test dataset).

Every op test base also wires up `InputValidationTestMixin` (`mloda/testing/feature_groups/data_operations/mixins/input_validation.py`) and overrides `input_validation_cases()`, declaring each of `multi_column_in_features`, `missing_source_column`, `empty_partition_by`, `missing_partition_by_column`, and `missing_order_by_column` as an `InputValidationCase` (feature name, context, `match`, and optionally a `table` for fresh test data), a reason string (skips with that reason), or `None` (does not apply to this op). The `multi_column_in_features` case also drives the inherited `test_mixin_empty_in_features` zero-in_features check, so declare it as an `InputValidationCase` (config-based) or a reason string to skip.

---

## Checklist

- [ ] One `OperationFeatureGroup` (mixins first, `OperationFeatureGroup` last) with `PREFIX_PATTERN`, `SUBTYPES`, and a shared loop that takes the framework step as a keyword-only callable.
- [ ] The loop calls `assert_source_columns_present(data, [...])` once per feature, before any per-backend dtype guard, and `assert_key_columns_present(data, ...)` for its partition, order and mask columns.
- [ ] PyArrow Implementation first; it is the reference.
- [ ] One `Implementation` per target framework (`feature_group`, `compute_framework`, one-line `calculate`), each respecting row-preserving if applicable; `supported` and `divergent` only where the framework genuinely cannot run or differs.
- [ ] `manifest.py` lists the group and every Implementation.
- [ ] Test base in `mloda/testing/.../{your_op}.py` with inherited test methods.
- [ ] `tests/test_base.py` subclasses `OperationGroupTestBase` with one `ImplSpec` per Implementation.
- [ ] One `tests/test_{framework}.py` per framework, importing the framework mixin.
- [ ] `supported_ops()` overrides match the Implementation's `supported`.
- [ ] Mask tests wired if the op consumes values that benefit from conditional inclusion.
- [ ] `OutputContractTestMixin` mixed in, with `output_contract_feature_set` overridden (and `output_contract_expected_row_count` overridden when the op changes the row count).
- [ ] `InputValidationTestMixin` wired with `input_validation_cases()` covering every kind.

---

## Related

- [Overview](01-overview.md) - Where this op fits in the category taxonomy.
- [Row-preserving contract](02-row-preserving-contract.md) - The invariant row-preserving ops must honor.
- [Reference implementation pattern](03-reference-implementation.md) - Why PyArrow goes first.
- [Supported ops per framework](04-supported-ops.md) - Mechanics of the `supported_ops` override.
- [Framework-Specific](../feature-group-patterns/09-framework-specific.md) - `OperationFeatureGroup`, `Implementation`, `supported` and `divergent`.
