"""Shared arithmetic test mixin: runs the pinned rows and the overflow rows through the backend under test."""

from __future__ import annotations

import math
from typing import Any, ClassVar

import pytest
from mloda.provider import DivergenceKind
from mloda.user import DataType

from mloda.testing.feature_groups.data_operations.base import divergent_subtypes
from mloda.testing.feature_groups.data_operations.helpers import (
    assert_values_with_nulls,
    calculate_with,
    make_feature_set,
)
from mloda.testing.feature_groups.data_operations.mixins.pinned_rows import PinnedRowTestMixin
from mloda.testing.feature_groups.data_operations.pinning import PinningRow, PinningTable
from mloda.testing.feature_groups.data_operations.row_preserving.arithmetic.pinning_rows import OverflowRow
from mloda.testing.typed_data import TypedTestData

OVERFLOW_MESSAGE = r"(?i)overflow|outside|out of range"


class ArithmeticPinningTestMixin(PinnedRowTestMixin):
    """Pinned rows and overflow rows of an arithmetic group; the host base names its table and rows."""

    _case_fixtures: ClassVar[dict[str, str]] = {"overflow_case": "overflow_cases"}

    @classmethod
    def pinning_table(cls) -> PinningTable:
        return cls.arithmetic_pinning_table()

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
    def overflow_cases(cls) -> dict[str, Any]:
        return {row.case: row for row in cls.arithmetic_overflow_rows()}

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

    def expected_values(self, row: PinningRow) -> dict[str, list[Any]]:
        """Apply only the backend's declared non-finite and signed-zero behavior."""
        expected = super().expected_values(row)
        op = self.feature_group_class().get_arithmetic_op(row.feature)
        data_type, values = row.expected[row.feature]
        is_float = data_type in (DataType.FLOAT, DataType.DOUBLE)
        nan_to_null = self.framework_nan_is_null()
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

        expected[row.feature] = [expect(v) for v in values]
        return expected

    def assert_pinned_values(self, row: PinningRow, column: str, actual: list[Any], expected: list[Any]) -> None:
        """Compare numeric values strictly after casting, including the sign of a zero result."""
        data_type = row.expected[column][0]
        cast = float if data_type in (DataType.FLOAT, DataType.DOUBLE) else int
        assert_values_with_nulls(actual, expected, strict=True, cast=cast)

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
