"""Shared arithmetic test mixin: runs the pinned rows and the overflow rows through the backend under test."""

from __future__ import annotations

import math
from typing import Any, ClassVar

import pytest
from mloda.provider import DivergenceKind
from mloda.user import DataType

from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase, divergent_subtypes
from mloda.testing.feature_groups.data_operations.helpers import (
    assert_values_with_nulls,
    calculate_with,
    make_feature_set,
)
from mloda.testing.feature_groups.data_operations.mixins.case_parametrization import CaseParametrizationTestMixin
from mloda.testing.feature_groups.data_operations.pinning import PinningTable
from mloda.testing.feature_groups.data_operations.row_preserving.arithmetic.pinning_rows import OverflowRow
from mloda.testing.typed_data import TypedTestData

OVERFLOW_MESSAGE = r"(?i)overflow|outside|out of range"


class ArithmeticPinningTestMixin(DataOpsTestBase, CaseParametrizationTestMixin):
    """Pinned rows and overflow rows of an arithmetic group; the host base names its table and rows."""

    _case_fixtures: ClassVar[dict[str, str]] = {
        "pinned_row_case": "pinned_row_cases",
        "overflow_case": "overflow_cases",
    }

    @classmethod
    def arithmetic_pinning_table(cls) -> PinningTable:
        """Hand-computed rows of the group under test."""
        raise NotImplementedError("Subclass must override arithmetic_pinning_table")

    @classmethod
    def arithmetic_overflow_rows(cls) -> tuple[OverflowRow, ...]:
        """Inputs whose integer result leaves its operands' range."""
        raise NotImplementedError("Subclass must override arithmetic_overflow_rows")

    @classmethod
    def keeps_zero_sign(cls) -> bool:
        """Whether the backend keeps the sign of a zero result."""
        return True

    @classmethod
    def pinned_row_cases(cls) -> dict[str, Any]:
        return {f"{i}-{row.case}": row for i, row in enumerate(cls.arithmetic_pinning_table().rows)}

    @classmethod
    def overflow_cases(cls) -> dict[str, Any]:
        return {row.case: row for row in cls.arithmetic_overflow_rows()}

    @classmethod
    def _nan_as_null_applies(cls, op: str) -> bool:
        """Whether a ``nan_as_null`` VARIANT record covers ``op``."""
        return any(
            record.kind is DivergenceKind.VARIANT
            and record.variant == "nan_as_null"
            and not record.outputs
            and (not record.subtypes or op in record.subtypes)
            for record in getattr(cls.implementation_class(), "divergences", ())
        )

    @classmethod
    def _divide_is_null_where_non_finite(cls) -> bool:
        """Whether a non-GAP record names ``divide`` (non-finite quotient comes back as None)."""
        return "divide" in divergent_subtypes(cls.implementation_class(), named_only=True)

    @classmethod
    def _overflow_is_a_gap(cls, op: str) -> bool:
        """Whether a GAP record names ``op``."""
        return any(
            record.kind is DivergenceKind.GAP and not record.outputs and op in record.subtypes
            for record in getattr(cls.implementation_class(), "divergences", ())
        )

    def test_mixin_pinned_row(self, pinned_row_case: str) -> None:
        """Rows are compared by value, so the result type is not checked; the sign of a zero result is."""
        row = self.pinned_row_cases()[pinned_row_case]
        op = self.feature_group_class().get_arithmetic_op(row.feature)
        data_type, values = row.expected[row.feature]
        is_float = data_type in (DataType.FLOAT, DataType.DOUBLE)
        nan_to_null = self._nan_as_null_applies(op)
        non_finite_to_null = op == "divide" and self._divide_is_null_where_non_finite()

        def expect(value: Any) -> Any:
            if value is None:
                return None
            if is_float:
                value = float(value)
                if (math.isnan(value) and nan_to_null) or (not math.isfinite(value) and non_finite_to_null):
                    return None
                return value if self.keeps_zero_sign() else value + 0.0
            return int(value)

        def cast(value: Any) -> Any:
            return float(value) if is_float else int(value)

        result = calculate_with(
            self.implementation_class(),
            self.create_test_data(TypedTestData(row.inputs)),
            make_feature_set(row.feature, **row.context),
        )
        assert_values_with_nulls(
            self.extract_column(result, row.feature), [expect(v) for v in values], strict=True, cast=cast
        )

    def test_mixin_integer_overflow(self, overflow_case: str) -> None:
        """An integer result outside its operands' range raises, unless a GAP record names the op."""
        row = self.overflow_cases()[overflow_case]
        op = self.feature_group_class().get_arithmetic_op(row.feature)
        length = len(next(iter(row.inputs.values()))[1])

        def run() -> list[Any]:
            result = calculate_with(
                self.implementation_class(),
                self.create_test_data(TypedTestData(row.inputs)),
                make_feature_set(row.feature, **row.context),
            )
            return self.extract_column(result, row.feature)

        if self._overflow_is_a_gap(op):
            assert len(run()) == length
            return
        with pytest.raises(Exception, match=OVERFLOW_MESSAGE):
            run()
