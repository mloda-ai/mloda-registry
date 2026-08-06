# Adding a New Data Operation

End-to-end recipe for introducing a new data operation: base class, framework implementations, test base, test mixins, and how to wire everything so the cross-framework comparison works.

**What**: The concrete steps to add a new operation category (or a new op inside an existing category) and have it tested across every supported framework.
**When**: You want a declarative transform that behaves identically on PyArrow, Pandas, Polars, DuckDB, and SQLite.
**Why**: The existing pattern enforces the row-preserving contract, reference-implementation comparison, and supported-ops skipping for free. Following it gets you coverage without reinventing test harnesses.
**Where**: Code in `mloda/community/feature_groups/data_operations/{your_category}/`, test bases in `mloda/testing/feature_groups/data_operations/{your_category}/`.
**How**: Follow the seven steps below in order. Every existing category (binning, window_aggregation, rank, offset, percentile, scalar_aggregate, scalar_arithmetic, point_arithmetic, frame_aggregate, string, aggregation) was built the same way.

---

## Step 1: Decide the category

- **Row-preserving**? Place it under `row_preserving/{your_op}/`. Read [the row-preserving contract](02-row-preserving-contract.md) first.
- **Row-reducing** (group aggregate)? Place it under `aggregation/` if it fits the `__{agg}_agg` naming; otherwise it probably wants its own folder at the top level of `data_operations/`.
- **Element-wise on strings**? Extend `string/` only if the new op has no parameters. Otherwise build a separate feature group.

The category decides your naming pattern. See [overview](01-overview.md) for the conventions.

---

## Step 2: Write the base class

File: `mloda/community/feature_groups/data_operations/{category}/{your_op}/base.py`.

```python
from typing import Any
from mloda.provider import FeatureGroup, FeatureSet


YOUR_OPS = {
    "op_a": "What op_a does",
    "op_b": "What op_b does",
}


class YourOpFeatureGroup(FeatureGroup):
    PREFIX_PATTERN = r".+__(op_a|op_b)$"

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        table = data
        for feature in features.features:
            source_col = cls._extract_source_features(feature)[0]
            op = cls._extract_op(feature)
            table = cls._compute(table, feature.name, source_col, op)
        return table

    @classmethod
    def _compute(cls, data: Any, feature_name: str, source_col: str, op: str) -> Any:
        raise NotImplementedError
```

The base class owns:

- The feature-name regex.
- The loop over `features.features`.
- Delegation to a per-framework `_compute` hook.

Source-column extraction, the ordering reads and the op-type accessors are inherited, not written. Mix in the one that matches your operation, always **last** in the bases (inserting it earlier shifts `__mro__.index()` for every ancestor behind it, and core reads that index as link specificity in `resolve_links`), and declare the attributes that drive it. Both rules are checked at class creation: a mixin listed before another base, or a driving attribute you never declared, raises `TypeError` on import.

| Mixin | Gives you | Declare |
|---|---|---|
| `SingleSourceMixin` | `_single_source_features`, `_single_source_input_features` | `MIN_IN_FEATURES`, `MAX_IN_FEATURES`, `SOURCE_LABEL`, `ENFORCE_MIN_IN_FEATURES`, `ENFORCE_MAX_IN_FEATURES`, `VALIDATE_IN_FEATURE_COUNT` |
| `PartitionedSourceMixin` | the above plus `_extract_partition_by` | `PARTITION_BY` |
| `OrderedSourceMixin` | the above plus `_extract_order_by` | `ORDER_BY`, `ORDER_BY_DEFAULTS_TO_SOURCE` |
| `OpTypeAccessorMixin` | `_extract_op_type`, `_resolve_op_type` | `_op_type_key()`, `OP_TYPE_LABEL` |

`SOURCE_LABEL` is only the noun the arity and ordering errors name your family with (unset, they name the class). What turns the checks on is `ENFORCE_MIN_IN_FEATURES` / `ENFORCE_MAX_IN_FEATURES`, and the bounds they read are core's `MIN_IN_FEATURES` / `MAX_IN_FEATURES`, so declare all four or the family accepts any number of source features. `ORDER_BY_DEFAULTS_TO_SOURCE` decides the other contract: leave it `False` and a missing `order_by` raises, set it `True` (sessionization) and the ordering falls back to the source column. `_op_type_key` is a classmethod rather than a constant so that a subclass overriding your family's option-key constant changes the key that is read.

Core owns `_extract_source_features` and `input_features` and resolves them before an appended mixin, so those two stay one-line delegators in your class:

```python
class YourOpFeatureGroup(RejectionReasonMixin, FeatureGroup, OrderedSourceMixin):
    MIN_IN_FEATURES = 1
    MAX_IN_FEATURES = 1

    SOURCE_LABEL = "your_op"
    ENFORCE_MIN_IN_FEATURES = True
    ENFORCE_MAX_IN_FEATURES = True

    PARTITION_BY = "partition_by"
    ORDER_BY = "order_by"

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return self._single_source_input_features(options, feature_name)

    @classmethod
    def _extract_source_features(cls, feature: Feature) -> list[str]:
        return cls._single_source_features(feature)
```

Existing bases to crib from: `row_preserving/datetime/base.py` or `row_preserving/scalar_aggregate/base.py` for the plain case, `row_preserving/ema/base.py` or `row_preserving/ffill/base.py` for the ordered one. `row_preserving/binning/base.py` and `row_preserving/sessionization/base.py` hand-write `input_features` where their contract differs from the mixin default, each with a comment saying why. `row_preserving/window_aggregation/base.py` predates the mixins and reads `partition_by` / `order_by` from `options` directly: crib its masks, not its extraction.

---

## Step 3: Write the PyArrow implementation first

PyArrow is the reference. Write it first; it defines correctness for everything else.

File: `{category}/{your_op}/pyarrow_{your_op}.py`.

```python
import pyarrow as pa
import pyarrow.compute as pc
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable
from .base import YourOpFeatureGroup


class PyArrowYourOp(YourOpFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type]:
        return {PyArrowTable}

    @classmethod
    def _compute(cls, data: pa.Table, feature_name: str, source_col: str, op: str) -> pa.Table:
        col = data.column(source_col)
        # ...op-specific columnar work...
        return data.append_column(feature_name, result_array)
```

Preserve row order. Preserve NULL. Handle the empty-table case.

---

## Step 4: Write the operation's test base

File: `mloda/testing/feature_groups/data_operations/{category}/{your_op}/{your_op}.py`.

```python
from typing import Any
import pytest
from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.community.feature_groups.data_operations.{category}.{your_op}.pyarrow_{your_op} import (
    PyArrowYourOp,
)


class YourOpTestBase(DataOpsTestBase):
    @classmethod
    def reference_implementation_class(cls) -> Any:
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

The test method calls `_compare_with_reference`, which runs both the implementation under test and PyArrow and asserts they match. Every framework's concrete test class inherits it.

One parametrized method per family, not one method per op: the explicit ids name the failing case (`test_cross_framework[op_b]`), and calling `_skip_if_unsupported` inside the body keeps the skip per case, so an unsupported op skips alone.

---

## Step 5: Add the other framework implementations

One file per framework, all subclassing `YourOpFeatureGroup`:

```text
{category}/{your_op}/
  base.py
  pyarrow_{your_op}.py
  pandas_{your_op}.py
  polars_lazy_{your_op}.py
  duckdb_{your_op}.py
  sqlite_{your_op}.py
```

Each class implements `_compute` using its native primitives. For row-preserving ops, respect the row-order invariant. If the native operation reorders (DuckDB `NTILE`), use the `ROW_NUMBER()` tag-and-restore pattern from [the row-preserving contract](02-row-preserving-contract.md).

---

## Step 6: Wire up the framework test classes

One file per framework, all inheriting from your test base and the framework's mixin:

```python
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

No test method bodies. Everything is inherited. If the framework cannot do one op, override `supported_ops` to return the subset; `_skip_if_unsupported` handles the skip.

---

## Step 7: Decide if it needs mask support

If the new op is an aggregate (or anything that consumes source values that a user might want to conditionally include), wire it up to `MaskTestMixin`:

```python
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

`MaskTestMixin` adds six inherited test methods covering equal, AND-combined, `is_in`, greater-than, fully-masked, and no-mask-baseline scenarios. See [Masking](../feature-group-patterns/25-masking.md) for the full user-facing spec.

---

## Checklist

- [ ] Base class with `PREFIX_PATTERN`, `calculate_feature`, and an abstract `_compute` hook.
- [ ] The matching extraction mixin appended last, with its driving attributes declared.
- [ ] PyArrow implementation first; it is the reference.
- [ ] Test base in `mloda/testing/.../{your_op}.py` with inherited test methods.
- [ ] One framework implementation per target framework, each respecting row-preserving if applicable.
- [ ] One `tests/test_{framework}.py` per framework, importing the framework mixin.
- [ ] `supported_ops()` overrides only where the framework genuinely cannot do the op.
- [ ] Mask tests wired if the op consumes values that benefit from conditional inclusion.

---

## Related

- [Overview](01-overview.md) - Where this op fits in the category taxonomy.
- [Row-preserving contract](02-row-preserving-contract.md) - The invariant row-preserving ops must honor.
- [Reference implementation pattern](03-reference-implementation.md) - Why PyArrow goes first.
- [Supported ops per framework](04-supported-ops.md) - Mechanics of the `supported_ops` override.
