# Adding a New Data Operation

End-to-end recipe for introducing a new data operation: one `OperationFeatureGroup`, one `Implementation` per framework, test base, test mixins, and how to wire everything so the cross-framework comparison works.

**What**: The concrete steps to add a new operation category (or a new op inside an existing category) and have it tested across every supported framework.
**When**: You want a declarative transform that behaves identically on every supported framework.
**Why**: The existing pattern enforces the row-preserving contract, reference-implementation comparison, and supported-ops skipping for free. Following it gets you coverage without reinventing test harnesses.
**Where**: Code in `mloda/community/feature_groups/data_operations/{row_preserving|row_changing}/{your_category}/`, test bases in the same layout under `mloda/testing/feature_groups/data_operations/`.
**How**: Follow the seven steps below in order (see [Overview](01-overview.md#the-operation-categories)).

---

## Step 1: Decide the category

- **Row-preserving**? Place it under `row_preserving/{your_op}/`. Read [the row-preserving contract](02-row-preserving-contract.md) first.
- **Row-reducing** (group aggregate)? Place it under `aggregation/` if it fits the `__{agg}_agg` naming; otherwise it probably wants its own folder at the top level of `data_operations/`. The group overrides [`replaces_rows()`](../feature-group-patterns/12-calculate-feature.md#returning-other-rows-replaces_rows) to return True and must return a new frame, never mutating its input. It must also use `KeyColumnInputsMixin` (as aggregation and resample do) so chains over its output plan.
- **Element-wise on strings**? Extend `string/` only if the new op has no parameters. Otherwise build a separate feature group.

The category decides your naming pattern. See [overview](01-overview.md) for the conventions.

---

## Step 2: Write the operation group

A new operation group declares `CONTRACT_VERSION = 1` in its class body.

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
    CONTRACT_VERSION = 1

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
            op = cls._extract_op(feature)  # your helper, modeled on _extract_string_op in string/base.py
            table = compute_your_op(table, feature.name, source_col, op)
        return table
```

The group owns:

- The base order: mixins first, `OperationFeatureGroup` last, so the mixin's matcher runs before the bare `FeatureGroup` one.
- The feature-name regex, `PROPERTY_MAPPING`, and `SUBTYPES` (the keys `supported` and `divergences` use).
- The shared loop as a classmethod that takes the framework step as a keyword-only callable. It defines no step itself and no `compute_framework_rule`: the frameworks are the ones its Implementations serve.
- Extraction of source column, operation, and any options from `Options(context=...)`.
- The shared `assert_source_columns_present` presence guard, called once per feature before any per-backend dtype guard.
- The shared `assert_key_columns_present` guard for `partition_by`, `order_by` (or `time_column`) and mask columns, called once per feature before the framework step.

Existing groups to crib from: `row_preserving/binning/base.py` (simple), `row_preserving/window_aggregation/base.py` (with `partition_by`/`order_by`/masks). They compose `FeatureChainParserMixin` to parse the suffix of the feature name; copy that detail verbatim from the closest existing group. Single-input ops use `_extract_single_source_feature`, multi-input ops `_extract_validated_source_features` (see `row_preserving/point_arithmetic/base.py`), so the MIN/MAX in_features and empty-operand checks live in the core helper, not a per-op override. Likewise, inherit `input_features`, `_extract_source_features` (a single-input op needs no override of it), and the match-time MIN/MAX check from the mixin instead of overriding them.

---

## Step 3: Write the PyArrow Implementation first

PyArrow is the first backend. Write it first; the kit reference (Step 4) defines correctness.

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

Every family has this shape; the reference each family compares against is listed in [Reference implementation pattern](03-reference-implementation.md#how-the-comparison-works). Rank and offset also declare `SUBTYPES` with parametric families (`ntile_N`, `top_N`, `bottom_N`; `lag_N`, `lead_N`, ...), which core collapses to the family name; lookup declares no `SUBTYPES`, so its Implementations declare no `supported` but some declare `divergences` (subtype-less `gap` records). See [Framework-Specific](../feature-group-patterns/09-framework-specific.md).

---

## Step 4: Write the operation's test bases

Every family test base inherits a check of the group's `CONTRACT_VERSION` against `contract_version()` (default 1); override it when the family's contract version moves.

File: `mloda/testing/feature_groups/data_operations/{category}/{your_op}/{your_op}.py`. These test bases import without pyarrow, so a PythonDict suite runs on a bare install. The reference comparison and the Arrow-only tests skip without `mloda-testing[pyarrow]`.

```py
from typing import Any
import pytest
from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase


class YourOpTestBase(DataOpsTestBase):
    @classmethod
    def reference_calculate(cls, table: Any, features: Any) -> Any:
        from mloda.testing.feature_groups.data_operations.{category}.{your_op}.reference import reference_your_op

        return reference_your_op(cls.feature_group_class(), table, features)

    @classmethod
    def supported_ops(cls) -> set[str]:
        return cls.declared_supported({"op_a", "op_b"})

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

The test method calls `_compare_with_reference`, which runs both the Implementation under test and the reference and asserts they match. Every framework's concrete test class inherits it.

Shipped modules under `mloda/testing/` may not import `mloda.community` or `mloda.enterprise` (a guard in `tests/test_end2end/test_dependency_direction.py` enforces it). A new operation therefore ships a kit-owned reference function `(group, table, features)` and overrides `reference_calculate` as aggregation, rank and offset do. The base no longer has `reference_implementation_class`; every test base overrides `reference_calculate` (see [How the comparison works](03-reference-implementation.md#how-the-comparison-works)). The reference also ships a `pinning_rows.py` of hand-computed rows (`PinningRow` and `PinningTable` from `mloda.testing.feature_groups.data_operations.pinning`) and one entry in the `PINNED` list of the pinning runner (runner: `mloda/community/feature_groups/data_operations/tests/test_reference_pinning.py`).

To run the table on each backend, mix `PinnedRowTestMixin` (`mloda.testing.feature_groups.data_operations.mixins.pinned_rows`) into the test base and return the table from `pinning_table()`; override `expected_values(row)` for declared divergences, `rejection_match(row)` to expect a backend's up-front `ValueError` on a row instead of a result, and `assert_pinned_values` for a stricter comparison (the arithmetic base overrides `expected_values` and `assert_pinned_values`, the datetime base `rejection_match`). A test base without it gets its `PINNED` rows run on every backend by `test_backend_pinning.py`, where a reproduced backend failure needs an entry in `_backend_pinning_limitations.py`.

One parametrized method per family, not one method per op: the explicit ids name the failing case (`test_cross_framework[op_b]`), and calling `_skip_if_unsupported` inside the body keeps the skip per case, so an unsupported op skips alone.

The group's shape gets its own test in `{category}/{your_op}/tests/test_base.py`: subclass `OperationGroupTestBase` (`mloda/testing/feature_groups/data_operations/operation_group.py`) and fill its hooks. It checks the base order, `SUBTYPES`, that the loop is on the group and the step only on the Implementations, each Implementation's framework pair, and the manifest. The claim hygiene (supported and divergences inside the universe, a reason on every divergence record, aliases, declines, `CONTRACT_VERSION`) is core's operation group shape check, run on the listed Implementations, so a listed one that serves no compute framework of the group fails; override `shape_probe()` when the subtype is resolver-based and an Implementation restricts `supported`:

```py
from mloda.testing.feature_groups.data_operations.operation_group import ImplSpec, OperationGroupTestBase

_ENGINE = "mloda_plugins.compute_framework.base_implementations"


class TestYourOpOperationGroup(OperationGroupTestBase):
    @classmethod
    def feature_group_class(cls) -> Any:
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

Each `ImplSpec.module` is the module that defines the Implementation. Set `dependency` for an optional backend: a missing dependency skips that spec, while a module that fails to import fails the test.

Optional hooks pin more: `universe()` the exact subtype universe (`frozenset()` for a group without `SUBTYPES`, see percentile/tests/test_base.py), `unresolved_probes()` names that resolve to no subtype, `mixins()` the mixins directly after the group, followed by `OperationFeatureGroup` and then `FeatureGroup`, `name_only_match()` the expected `NAME_ONLY_MATCH`, `alias_pairs()`, and, on `DataOpsIntegrationTestBase` (`mloda.testing.feature_groups.data_operations.integration`), `group_options()`, `data_access_collection()` (credentials a `group_options()` handle names) and `missing_config_feature_name()` (the name `test_match_rejects_missing_config` expects not to match without config; default `primary_feature_name()`).

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

Each sets `feature_group` and `compute_framework`, makes `calculate` the one-liner through `group.apply_your_op`, and implements `_compute_your_op` with its native primitives. For row-preserving ops, respect the row-order invariant. SQL `window()` keeps input order, so call it directly; DuckDB implementations wrap only a non-window reordering step (raw SQL, joins) in `preserve_row_order` (see [the row-preserving contract](02-row-preserving-contract.md)).

A framework that cannot run an op, or differs from the reference on one, declares `supported` / `divergences` on its Implementation (see [Framework-Specific](../feature-group-patterns/09-framework-specific.md#complete-example)).

Reject an op the framework cannot run natively through `supported`; do not compute it in Python inside the Implementation. Examples: `aggregation/sqlite_aggregation.py` (both), `point_arithmetic/sqlite_point_arithmetic.py` (`divergences` only).

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
from mloda.testing.feature_groups.data_operations.mixins.frameworks import PandasTestMixin
from mloda.testing.feature_groups.data_operations.{category}.{your_op}.{your_op} import (
    YourOpTestBase,
)


class TestPandasYourOp(PandasTestMixin, YourOpTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PandasYourOp
```

Base order: kit test mixins first, then the framework adapter mixin, then the family test base (or `DataOpsTestBase`) last; another order fails with an MRO error or leaves the class abstract so pytest skips it. No test method bodies; everything is inherited. `implementation_class` returns the Implementation, and the test base reads its `supported`, so no skip override is needed. To also pin the match-time capability per framework, mix in `CapabilityHookTestMixin` (`mloda.testing.feature_groups.data_operations.mixins.capability`; see `aggregation/tests/test_sqlite.py`). Shared pieces live beside `DataOpsTestBase`: `MatchValidationTestBase` and `ScalarArityTestBase` (`mloda.testing.feature_groups.data_operations.match_validation`), `OperationGroupIntegrationTestMixin` (`mloda.testing.feature_groups.data_operations.mixins.operation_group_integration`, for the integration class), `SingleValueStdVarTestMixin` (`mloda.testing.feature_groups.data_operations.mixins.single_value_std_var`, used by aggregation, window aggregation and frame aggregate) and the tie layouts of `mloda.testing.feature_groups.data_operations.tie_layouts` (used by offset and window aggregation tied-order tests).

From typed data, `PandasTestMixin` stores an `INT64` column with nulls as float64. To also run the suite on pandas nullable ints (`Int64` and friends), add a second class that swaps in `PandasNullableIntTestMixin`, e.g. `class TestPandasYourOpNullableInt(PandasNullableIntTestMixin, TestPandasYourOp)`, and reset any flags the float64 class sets for lossy ints. Likewise `PandasObjectStringTestMixin` / `PandasPythonStringTestMixin` run the suite on object / python string storage. New SQLite ops also add a `SqliteViewTestMixin` class so the suite also runs on a view input (every feature after the first sees one).

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

If the new op has a documented NaN policy (see [the reference-implementation guide](03-reference-implementation.md#when-pyarrow-has-no-kernel-or-is-the-outlier)), wire it up to `NanPolicyTestMixin` (`mloda/testing/feature_groups/data_operations/mixins/nan_policy.py`) instead of writing a bespoke `test_nan_policy_*` method. Override `nan_policy_cases` (case id -> expected value, required) and `nan_policy_feature_name` (case id -> feature name, required); `nan_policy_agg_type` defaults to the case id, `nan_policy_feature_set` defaults to partitioning by `"grp"`, `nan_policy_divergent_cases` defaults to no divergences, and `nan_policy_extract_values` defaults to a plain list column. A divergent case only applies for a concrete framework test class that names its agg type (via `nan_policy_agg_type`) in that class's `nan_divergent_agg_types()`, which by default is the subtypes the Implementation's shipped `divergences` records name explicitly (a record without subtypes is not counted; empty for a plain FeatureGroup subclass); a framework with no divergence for the op does not need to override `nan_policy_divergent_cases` at all. The default skip hook calls `_skip_if_unsupported`, which needs a `supported_*` set on the base (e.g. `supported_agg_types`); ops without one (resample, percentile) must override `nan_policy_skip_if_unsupported` instead.

A new case-parametrized test mixin inherits `CaseParametrizationTestMixin` (`mloda/testing/feature_groups/data_operations/mixins/case_parametrization.py`) and declares its fixture in `_case_fixtures`, rather than defining its own `pytest_generate_tests`; `mixins/tests/test_mixin_isolation.py` enforces this structurally.

Every test base also mixes in `OutputContractTestMixin` (`mloda/testing/feature_groups/data_operations/mixins/output_contract.py`), which covers the result-type, row-count, and new-column checks that used to be hand-written per base. Override `output_contract_feature_set` (an instance method) to return the `FeatureSet` to exercise, and `output_contract_expected_row_count` when the op changes the row count (return the expected count for the test dataset).

If the group's `return_data_type_rule` declares a type for any op, the test base also mixes in `ResultTypeTestMixin` (`mloda/testing/feature_groups/data_operations/mixins/result_type.py`) and overrides `result_type_cases()` (case id -> one-feature `FeatureSet` per typed op). It runs core's full shape check per op on every framework and fails on any finding (output type, zero rows, changed input columns, matching, determinism). Only a `type` record excuses a type mismatch, and a `gap` record only an unresolved type; a zero-row retype the adapter itself makes to an input column is ignored. Each case must read a column that holds a null in `result_type_data()` (default `source_data()`) and must be name-based, since an option-based request matches any name.

Every op test base also wires up `InputValidationTestMixin` (`mloda/testing/feature_groups/data_operations/mixins/input_validation.py`) and overrides `input_validation_cases()`, declaring each of `multi_column_in_features`, `missing_source_column`, `empty_partition_by`, `missing_partition_by_column`, and `missing_order_by_column` as an `InputValidationCase` (feature name, context, `match`, and optionally `data` (a `TypedTestData`) for fresh test data; a fixture overrides `source_data()`), a reason string (skips with that reason), or `None` (does not apply to this op). The `multi_column_in_features` case also drives the inherited `test_mixin_empty_in_features` zero-in_features check, so declare it as an `InputValidationCase` (config-based) or a reason string to skip.

---

## Related

- [Overview](01-overview.md) - Where this op fits in the category taxonomy.
- [Row-preserving contract](02-row-preserving-contract.md) - The invariant row-preserving ops must honor.
- [Reference implementation pattern](03-reference-implementation.md) - Where the reference lives.
- [Supported ops per framework](04-supported-ops.md) - How an Implementation's `supported` is read by the test bases.
- [Framework-Specific](../feature-group-patterns/09-framework-specific.md) - `OperationFeatureGroup`, `Implementation`, `supported` and `divergences`.
