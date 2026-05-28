"""Shared test base class and expected values for two-column point arithmetic tests.

Each test verifies that an element-wise arithmetic operation (add, subtract,
multiply, divide) is computed between two source columns. Null values in
either source column propagate to None in the output.

Expected values are computed from the canonical 12-row dataset in
``DataOperationsTestDataCreator``, using the ``value_int`` and ``amount`` columns
as ``col_a`` and ``col_b`` respectively.

value_int = [10, -5, 0, 20, None, 50, 30, 60, 15, 15, 40, -10]
amount    = [100.0, None, 250.0, 75.0, 300.0, 0.0, 150.0, None, 50.0, 200.0, 125.0, 80.0]
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda.user import Feature


# ---------------------------------------------------------------------------
# Source columns and expected values (module-level constants)
# ---------------------------------------------------------------------------

VALUE_INT: list[int | None] = [10, -5, 0, 20, None, 50, 30, 60, 15, 15, 40, -10]
AMOUNT: list[float | None] = [100.0, None, 250.0, 75.0, 300.0, 0.0, 150.0, None, 50.0, 200.0, 125.0, 80.0]


def _apply2(values_a: list[Any], values_b: list[Any], fn: Any) -> list[Any]:
    """Apply a binary function element-wise; propagate None when either operand is None."""
    out: list[Any] = []
    for a, b in zip(values_a, values_b):
        if a is None or b is None:
            out.append(None)
        else:
            out.append(fn(a, b))
    return out


def _expected_divide(*, zero_to_null: bool) -> list[Any]:
    """Element-wise divide expected values.

    For the canonical (VALUE_INT, AMOUNT) pair, row 5 is ``50 / 0.0``.
    PyArrow/Pandas/Polars/DuckDB cast to float and propagate IEEE-754
    inf/nan; SQLite has no IEEE-754 storage and returns NULL instead.

    Set ``zero_to_null=True`` for SQLite expectations, ``False`` for the
    other four backends.
    """
    out: list[Any] = []
    for a, b in zip(VALUE_INT, AMOUNT):
        if a is None or b is None:
            out.append(None)
        elif b == 0:
            if zero_to_null:
                out.append(None)
            else:
                if a == 0:
                    out.append(float("nan"))
                else:
                    out.append(math.copysign(float("inf"), a))
        else:
            out.append(a / b)
    return out


# Expected results for the canonical column pair
EXPECTED_ADD: list[Any] = _apply2(VALUE_INT, AMOUNT, lambda a, b: a + b)
EXPECTED_SUBTRACT: list[Any] = _apply2(VALUE_INT, AMOUNT, lambda a, b: a - b)
EXPECTED_MULTIPLY: list[Any] = _apply2(VALUE_INT, AMOUNT, lambda a, b: a * b)
EXPECTED_DIVIDE_INF: list[Any] = _expected_divide(zero_to_null=False)
EXPECTED_DIVIDE_NULL_FOR_ZERO: list[Any] = _expected_divide(zero_to_null=True)


# ---------------------------------------------------------------------------
# Reusable test base class
# ---------------------------------------------------------------------------


class PointArithmeticTestBase(DataOpsTestBase):
    """Abstract base class for two-column point arithmetic framework tests."""

    ALL_OPS = {"add", "subtract", "multiply", "divide"}

    @classmethod
    def supported_ops(cls) -> set[str]:
        """Override to restrict supported operations for a framework."""
        return cls.ALL_OPS

    @classmethod
    def reference_implementation_class(cls) -> Any:
        from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.pyarrow_point_arithmetic import (
            PyArrowPointArithmetic,
        )

        return PyArrowPointArithmetic

    # -- Source-column dtype contract (shared across backends) ---------------

    @classmethod
    def detects_non_numeric_source(cls) -> set[str]:
        """Types of non-numeric source columns the backend can reject.

        Default is ``{"string", "boolean"}``. SQLite overrides to ``{"string"}``
        because ``SqliteRelation.from_arrow`` stores booleans with INTEGER
        affinity and they cannot be distinguished from int64 at the relation
        level.
        """
        return {"string", "boolean"}

    @classmethod
    def divide_zero_propagates_inf(cls) -> bool:
        """Whether divide-by-zero produces IEEE-754 inf/nan or NULL.

        PyArrow / Pandas / Polars / DuckDB cast operands to float and
        propagate IEEE-754 special values. SQLite has no IEEE-754 storage
        and overrides this to ``False`` (NULL for divide-by-zero rows).
        """
        return True

    # -- Concrete tests -----------------------------------------------------

    def test_add(self) -> None:
        """value_int + amount, null in either operand propagates."""
        fs = make_feature_set("value_int&amount__add_point")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int&amount__add_point")
        for actual, expected in zip(result_col, EXPECTED_ADD):
            if expected is None:
                assert actual is None
            else:
                assert actual == pytest.approx(expected, rel=1e-6)

    def test_subtract(self) -> None:
        """value_int - amount, null propagates."""
        fs = make_feature_set("value_int&amount__subtract_point")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int&amount__subtract_point")
        for actual, expected in zip(result_col, EXPECTED_SUBTRACT):
            if expected is None:
                assert actual is None
            else:
                assert actual == pytest.approx(expected, rel=1e-6)

    def test_multiply(self) -> None:
        """value_int * amount, null propagates."""
        fs = make_feature_set("value_int&amount__multiply_point")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int&amount__multiply_point")
        for actual, expected in zip(result_col, EXPECTED_MULTIPLY):
            if expected is None:
                assert actual is None
            else:
                assert actual == pytest.approx(expected, rel=1e-6)

    def test_divide(self) -> None:
        """value_int / amount on non-zero rows, null propagates.

        Row 5 (50 / 0.0) is excluded here: the divide-by-zero contract is
        verified separately in ``test_divide_by_zero_returns_inf_or_null_per_backend``.
        """
        fs = make_feature_set("value_int&amount__divide_point")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int&amount__divide_point")
        for i, (actual, expected) in enumerate(zip(result_col, EXPECTED_DIVIDE_INF)):
            if i == 5:
                # Divide-by-zero row, handled by dedicated test.
                continue
            if expected is None:
                assert actual is None
            else:
                assert actual == pytest.approx(expected, rel=1e-6)

    def test_output_rows_equal_input_rows(self) -> None:
        fs = make_feature_set("value_int&amount__add_point")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert self.get_row_count(result) == 12

    def test_result_has_correct_type(self) -> None:
        fs = make_feature_set("value_int&amount__multiply_point")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())

    def test_new_column_added(self) -> None:
        fs = make_feature_set("value_int&amount__add_point")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int&amount__add_point")
        assert len(result_col) == 12

    def test_null_propagates_when_col_a_is_null(self) -> None:
        """A null in col_a propagates to None in the result regardless of op.

        Row index 4 of value_int is None.
        """
        fs = make_feature_set("value_int&amount__multiply_point")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int&amount__multiply_point")
        assert result_col[4] is None

    def test_null_propagates_when_col_b_is_null(self) -> None:
        """A null in col_b propagates to None in the result regardless of op.

        Row index 1 of amount is None.
        """
        fs = make_feature_set("value_int&amount__add_point")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int&amount__add_point")
        assert result_col[1] is None

    def test_unsupported_operation_raises(self) -> None:
        """An unrecognized arithmetic operation must raise ValueError."""
        feature = Feature(
            "value_int&amount__bogus_point",
            options=Options(),
        )
        fs = FeatureSet()
        fs.add(feature)
        with pytest.raises(ValueError, match="[Uu]nsupported|[Cc]ould not extract"):
            self.implementation_class().calculate_feature(self.test_data, fs)

    # -- Option-based configuration -----------------------------------------

    def test_option_based_add(self) -> None:
        """Option-based configuration (not string pattern) produces the same result."""
        feature = Feature(
            "my_custom_add",
            options=Options(
                context={
                    "arithmetic_op": "add",
                    "in_features": ["value_int", "amount"],
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "my_custom_add")
        for actual, expected in zip(result_col, EXPECTED_ADD):
            if expected is None:
                assert actual is None
            else:
                assert actual == pytest.approx(expected, rel=1e-6)
        assert len(result_col) == 12

    def test_option_based_subtract(self) -> None:
        """Option-based subtract produces the correct values."""
        feature = Feature(
            "my_diff",
            options=Options(
                context={
                    "arithmetic_op": "subtract",
                    "in_features": ["value_int", "amount"],
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "my_diff")
        for actual, expected in zip(result_col, EXPECTED_SUBTRACT):
            if expected is None:
                assert actual is None
            else:
                assert actual == pytest.approx(expected, rel=1e-6)

    def test_option_based_multiply(self) -> None:
        """Option-based multiply produces the correct values."""
        feature = Feature(
            "my_product",
            options=Options(
                context={
                    "arithmetic_op": "multiply",
                    "in_features": ["value_int", "amount"],
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "my_product")
        for actual, expected in zip(result_col, EXPECTED_MULTIPLY):
            if expected is None:
                assert actual is None
            else:
                assert actual == pytest.approx(expected, rel=1e-6)

    # -- In-feature count enforcement at compute time -----------------------

    def test_too_few_in_features_rejected_at_calculate(self) -> None:
        """calculate_feature must reject features with fewer than 2 in_features."""
        feature = Feature(
            "bad_too_few",
            options=Options(
                context={
                    "arithmetic_op": "add",
                    "in_features": ["value_int"],
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        with pytest.raises(ValueError, match="at least 2"):
            self.implementation_class().calculate_feature(self.test_data, fs)

    def test_too_many_in_features_rejected_at_calculate(self) -> None:
        """calculate_feature must reject features with more than 2 in_features."""
        feature = Feature(
            "bad_too_many",
            options=Options(
                context={
                    "arithmetic_op": "add",
                    "in_features": ["value_int", "amount", "value_float"],
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        with pytest.raises(ValueError, match="at most 2"):
            self.implementation_class().calculate_feature(self.test_data, fs)

    # -- Source-column dtype enforcement ------------------------------------

    def test_string_col_a_rejected(self) -> None:
        """A non-numeric (string) col_a must raise ValueError naming the column."""
        if "string" not in self.detects_non_numeric_source():
            pytest.skip("Backend cannot distinguish string source columns at relation level")
        import re

        fs = make_feature_set("name&value_int__add_point")
        with pytest.raises(ValueError, match=r"(?i)numeric") as exc_info:
            self.implementation_class().calculate_feature(self.test_data, fs)
        assert re.search(r"['\"]name['\"]", str(exc_info.value)), (
            f"Expected source column 'name' to be named (quoted) in the error message, got: {exc_info.value!r}"
        )

    def test_string_col_b_rejected(self) -> None:
        """A non-numeric (string) col_b must raise ValueError naming the column."""
        if "string" not in self.detects_non_numeric_source():
            pytest.skip("Backend cannot distinguish string source columns at relation level")
        import re

        fs = make_feature_set("value_int&name__add_point")
        with pytest.raises(ValueError, match=r"(?i)numeric") as exc_info:
            self.implementation_class().calculate_feature(self.test_data, fs)
        assert re.search(r"['\"]name['\"]", str(exc_info.value)), (
            f"Expected source column 'name' to be named (quoted) in the error message, got: {exc_info.value!r}"
        )

    def test_boolean_source_column_rejected_col_a(self) -> None:
        """A boolean col_a must raise ValueError naming the column."""
        if "boolean" not in self.detects_non_numeric_source():
            pytest.skip("Backend cannot distinguish boolean source columns at relation level")
        fs = make_feature_set("is_active&value_int__add_point")
        with pytest.raises(ValueError, match=r"(?i)numeric") as exc_info:
            self.implementation_class().calculate_feature(self.test_data, fs)
        assert "is_active" in str(exc_info.value)

    def test_boolean_source_column_rejected_col_b(self) -> None:
        """A boolean col_b must raise ValueError naming the column."""
        if "boolean" not in self.detects_non_numeric_source():
            pytest.skip("Backend cannot distinguish boolean source columns at relation level")
        fs = make_feature_set("value_int&is_active__add_point")
        with pytest.raises(ValueError, match=r"(?i)numeric") as exc_info:
            self.implementation_class().calculate_feature(self.test_data, fs)
        assert "is_active" in str(exc_info.value)

    # -- Cross-framework comparison -----------------------------------------

    def _compare_arithmetic_with_reference(self, feature_name: str) -> None:
        """Compute the feature on this framework and on the reference; assert equal."""
        fs = make_feature_set(feature_name)
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
        self._compare_arithmetic_with_reference("value_int&amount__add_point")

    def test_cross_framework_subtract(self) -> None:
        self._compare_arithmetic_with_reference("value_int&amount__subtract_point")

    def test_cross_framework_multiply(self) -> None:
        self._compare_arithmetic_with_reference("value_int&amount__multiply_point")

    # -- Divide-by-zero per-backend semantics -------------------------------

    def test_divide_by_zero_returns_inf_or_null_per_backend(self) -> None:
        """For the canonical (value_int, amount) pair, row 5 is 50 / 0.0.

        Four backends (PyArrow / Pandas / Polars / DuckDB) cast operands to
        float and produce IEEE-754 ``+inf`` (positive numerator) at row 5.
        SQLite has no IEEE-754 storage and returns NULL.
        """
        fs = make_feature_set("value_int&amount__divide_point")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_col = self.extract_column(result, "value_int&amount__divide_point")

        expected = EXPECTED_DIVIDE_INF if self.divide_zero_propagates_inf() else EXPECTED_DIVIDE_NULL_FOR_ZERO

        for i, (actual, exp) in enumerate(zip(result_col, expected)):
            if exp is None:
                assert actual is None, f"row {i}: expected None got {actual}"
            elif isinstance(exp, float) and math.isnan(exp):
                assert actual is not None and isinstance(actual, float) and math.isnan(actual), (
                    f"row {i}: expected nan got {actual}"
                )
            elif isinstance(exp, float) and math.isinf(exp):
                assert actual is not None and isinstance(actual, float) and math.isinf(actual), (
                    f"row {i}: expected inf got {actual}"
                )
                # Verify the sign of inf matches.
                assert math.copysign(1.0, actual) == math.copysign(1.0, exp), (
                    f"row {i}: inf sign mismatch (expected {exp}, got {actual})"
                )
            else:
                assert actual == pytest.approx(exp, rel=1e-6), f"row {i}: {actual} != {exp}"

    def test_divide_by_integer_constant_returns_float_per_row(self) -> None:
        """Regression guard: int/int divide must still produce float results.

        Use ``value_int`` as both operands. Each non-null row should be a
        Python float (or compatible) after the divide cast, never an int.
        """
        fs = make_feature_set("value_int&value_int__divide_point")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_col = self.extract_column(result, "value_int&value_int__divide_point")

        for i, value in enumerate(result_col):
            if VALUE_INT[i] is None or VALUE_INT[i] == 0:
                # Null and 0/0 rows are excluded from the float-type assertion:
                # 0/0 is handled by the divide-by-zero contract elsewhere.
                continue
            assert isinstance(value, float), f"row {i}: expected float got {type(value).__name__} ({value!r})"
