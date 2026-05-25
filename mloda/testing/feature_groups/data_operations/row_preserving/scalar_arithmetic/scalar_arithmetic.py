"""Shared test base class and expected values for single-column scalar arithmetic tests.

Each test verifies that an element-wise arithmetic operation (add, subtract,
multiply, divide) is computed between a source column and a numeric constant.
Null values in the source column remain null in the output.

Expected values are computed from the canonical 12-row dataset in
DataOperationsTestDataCreator, using the 'value_int' column.

value_int = [10, -5, 0, 20, None, 50, 30, 60, 15, 15, 40, -10]
"""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda.user import Feature


# ---------------------------------------------------------------------------
# Source column and expected values (module-level constants)
# ---------------------------------------------------------------------------

VALUE_INT: list[int | None] = [10, -5, 0, 20, None, 50, 30, 60, 15, 15, 40, -10]


def _apply(values: list[int | None], fn: Any) -> list[Any]:
    return [None if v is None else fn(v) for v in values]


# Expected results for canonical test constants
EXPECTED_ADD_5: list[Any] = _apply(VALUE_INT, lambda v: v + 5)
EXPECTED_SUBTRACT_10: list[Any] = _apply(VALUE_INT, lambda v: v - 10)
EXPECTED_MULTIPLY_2: list[Any] = _apply(VALUE_INT, lambda v: v * 2)
EXPECTED_DIVIDE_2: list[Any] = _apply(VALUE_INT, lambda v: v / 2.0)


# ---------------------------------------------------------------------------
# Reusable test base class
# ---------------------------------------------------------------------------


class ScalarArithmeticTestBase(DataOpsTestBase):
    """Abstract base class for scalar arithmetic framework tests."""

    ALL_OPS = {"add", "subtract", "multiply", "divide"}

    @classmethod
    def supported_ops(cls) -> set[str]:
        """Override to restrict supported operations for a framework."""
        return cls.ALL_OPS

    @classmethod
    def reference_implementation_class(cls) -> Any:
        from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.pyarrow_scalar_arithmetic import (
            PyArrowScalarArithmetic,
        )

        return PyArrowScalarArithmetic

    # -- Concrete tests -----------------------------------------------------

    def test_add_constant(self) -> None:
        """value_int + 5, null preserved."""
        fs = make_feature_set("value_int__add_constant", constant=5)
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__add_constant")
        for actual, expected in zip(result_col, EXPECTED_ADD_5):
            if expected is None:
                assert actual is None
            else:
                assert actual == pytest.approx(expected, rel=1e-6)

    def test_subtract_constant(self) -> None:
        """value_int - 10, null preserved."""
        fs = make_feature_set("value_int__subtract_constant", constant=10)
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__subtract_constant")
        for actual, expected in zip(result_col, EXPECTED_SUBTRACT_10):
            if expected is None:
                assert actual is None
            else:
                assert actual == pytest.approx(expected, rel=1e-6)

    def test_multiply_constant(self) -> None:
        """value_int * 2, null preserved."""
        fs = make_feature_set("value_int__multiply_constant", constant=2)
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__multiply_constant")
        for actual, expected in zip(result_col, EXPECTED_MULTIPLY_2):
            if expected is None:
                assert actual is None
            else:
                assert actual == pytest.approx(expected, rel=1e-6)

    def test_divide_constant(self) -> None:
        """value_int / 2.0, null preserved."""
        fs = make_feature_set("value_int__divide_constant", constant=2.0)
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__divide_constant")
        for actual, expected in zip(result_col, EXPECTED_DIVIDE_2):
            if expected is None:
                assert actual is None
            else:
                assert actual == pytest.approx(expected, rel=1e-6)

    def test_output_rows_equal_input_rows(self) -> None:
        fs = make_feature_set("value_int__add_constant", constant=5)
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert self.get_row_count(result) == 12

    def test_result_has_correct_type(self) -> None:
        fs = make_feature_set("value_int__multiply_constant", constant=2)
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())

    def test_new_column_added(self) -> None:
        fs = make_feature_set("value_int__add_constant", constant=5)
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__add_constant")
        assert len(result_col) == 12

    def test_null_values_preserved(self) -> None:
        """A null in the source must remain null in the result, regardless of op."""
        fs = make_feature_set("value_int__multiply_constant", constant=7)
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__multiply_constant")
        # Row index 4 of value_int is None.
        assert result_col[4] is None

    def test_divide_by_zero_raises(self) -> None:
        """Dividing by zero must raise ValueError."""
        fs = make_feature_set("value_int__divide_constant", constant=0)
        with pytest.raises(ValueError, match="[Dd]ivide|[Zz]ero|0"):
            self.implementation_class().calculate_feature(self.test_data, fs)

    def test_missing_constant_raises(self) -> None:
        """If Options has no constant key, calculate_feature must raise ValueError."""
        feature = Feature("value_int__add_constant", options=Options())
        fs = FeatureSet()
        fs.add(feature)
        with pytest.raises(ValueError, match="constant"):
            self.implementation_class().calculate_feature(self.test_data, fs)

    def test_unsupported_operation_raises(self) -> None:
        """An unrecognized arithmetic operation must raise ValueError."""
        feature = Feature(
            "value_int__bogus_constant",
            options=Options(context={"constant": 5}),
        )
        fs = FeatureSet()
        fs.add(feature)
        with pytest.raises(ValueError, match="[Uu]nsupported|[Cc]ould not extract"):
            self.implementation_class().calculate_feature(self.test_data, fs)

    def test_option_based_add(self) -> None:
        """Option-based configuration (not string pattern) produces the same result."""
        feature = Feature(
            "my_custom_add",
            options=Options(
                context={
                    "arithmetic_op": "add",
                    "in_features": "value_int",
                    "constant": 5,
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "my_custom_add")
        for actual, expected in zip(result_col, EXPECTED_ADD_5):
            if expected is None:
                assert actual is None
            else:
                assert actual == pytest.approx(expected, rel=1e-6)
        assert len(result_col) == 12

    def test_option_based_multiply(self) -> None:
        """Option-based multiply produces the correct values."""
        feature = Feature(
            "my_doubled",
            options=Options(
                context={
                    "arithmetic_op": "multiply",
                    "in_features": "value_int",
                    "constant": 2,
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "my_doubled")
        for actual, expected in zip(result_col, EXPECTED_MULTIPLY_2):
            if expected is None:
                assert actual is None
            else:
                assert actual == pytest.approx(expected, rel=1e-6)

    def test_multi_column_in_features_rejected_at_calculate(self) -> None:
        """calculate_feature must reject features with multiple in_features.

        Scalar arithmetic supports exactly one source column.
        """
        feature = Feature(
            "bad_multi",
            options=Options(
                context={
                    "arithmetic_op": "add",
                    "in_features": ["value_int", "value_float"],
                    "constant": 5,
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        with pytest.raises(ValueError, match="at most 1"):
            self.implementation_class().calculate_feature(self.test_data, fs)

    # ---- Source-column dtype contract (shared across backends) ----

    @classmethod
    def detects_non_numeric_source(cls) -> set[str]:
        """Types of non-numeric source columns the backend can reject.

        Default is ``{"string", "boolean"}``. Backends override when a dtype
        cannot be distinguished from numeric at the relation level (e.g.
        SQLite collapses booleans to INTEGER affinity through ``from_arrow``).
        """
        return {"string", "boolean"}

    def test_string_source_column_rejected(self) -> None:
        """A non-numeric (string) source column must raise ValueError with a clear
        message naming the column."""
        if "string" not in self.detects_non_numeric_source():
            pytest.skip("Backend cannot distinguish string source columns at relation level")
        import re

        fs = make_feature_set("name__add_constant", constant=5)
        with pytest.raises(ValueError, match=r"(?i)numeric") as exc_info:
            self.implementation_class().calculate_feature(self.test_data, fs)
        # Quoted form keeps the assertion from accidentally matching against the
        # ``name__add_constant`` feature name. Most existing mloda errors quote column
        # names via ``!r`` or ``repr``.
        assert re.search(r"['\"]name['\"]", str(exc_info.value)), (
            f"Expected source column 'name' to be named (quoted) in the error message, got: {exc_info.value!r}"
        )

    def test_boolean_source_column_rejected(self) -> None:
        """A boolean source column must raise ValueError."""
        if "boolean" not in self.detects_non_numeric_source():
            pytest.skip("Backend cannot distinguish boolean source columns at relation level")
        fs = make_feature_set("is_active__add_constant", constant=5)
        with pytest.raises(ValueError, match=r"(?i)numeric") as exc_info:
            self.implementation_class().calculate_feature(self.test_data, fs)
        assert "is_active" in str(exc_info.value)

    # -- Cross-framework comparison -----------------------------------------

    def _compare_arithmetic_with_reference(
        self,
        feature_name: str,
        constant: int | float,
    ) -> None:
        """Compute the feature on this framework and on the reference; assert equal."""
        fs = make_feature_set(feature_name, constant=constant)
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        ref = self.reference_implementation_class().calculate_feature(self._arrow_table, fs)

        result_col = self.extract_column(result, feature_name)
        # Reference is a pyarrow.Table.
        ref_col = list(ref.column(feature_name).to_pylist())

        assert len(result_col) == len(ref_col)
        for i, (ref_val, fw_val) in enumerate(zip(ref_col, result_col)):
            if ref_val is None:
                assert fw_val is None, f"row {i}: expected None, got {fw_val}"
            else:
                assert fw_val == pytest.approx(ref_val, rel=1e-6), f"row {i}: {fw_val} != {ref_val}"

    def test_cross_framework_add(self) -> None:
        self._compare_arithmetic_with_reference("value_int__add_constant", 5)

    def test_cross_framework_subtract(self) -> None:
        self._compare_arithmetic_with_reference("value_int__subtract_constant", 10)

    def test_cross_framework_multiply(self) -> None:
        self._compare_arithmetic_with_reference("value_int__multiply_constant", 2)

    def test_cross_framework_divide(self) -> None:
        self._compare_arithmetic_with_reference("value_int__divide_constant", 2.0)

    def test_divide_by_integer_constant_returns_float(self) -> None:
        """Cross-backend: dividing by an INT constant must still produce float results.

        Regression guard: PyArrow's pc.divide truncates int÷int to int; SQL
        backends cast to DOUBLE/REAL; Pandas/Polars use Python /. All five
        must agree on the float-output semantics.
        """
        fs = make_feature_set("value_int__divide_constant", constant=3)  # int constant
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__divide_constant")
        # value_int = [10, -5, 0, 20, None, 50, 30, 60, 15, 15, 40, -10]
        # Expected: each non-null v / 3 as float
        expected = [None if v is None else v / 3 for v in VALUE_INT]
        for actual, exp in zip(result_col, expected):
            if exp is None:
                assert actual is None
            else:
                assert actual == pytest.approx(exp, rel=1e-6), f"divide-by-3 expected {exp} got {actual}"
