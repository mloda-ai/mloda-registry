"""Shared test base for scalar arithmetic; expected values come from the reference."""

from __future__ import annotations

import re
from typing import Any

import pyarrow as pa
import pytest
from mloda.provider import FeatureSet
from mloda.user import Feature, Options

from mloda.testing.data_creator.base import DataOperationsTestDataCreator
from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.helpers import (
    assert_values_with_nulls,
    calculate_with,
    make_feature_set,
    typed_data_without,
)
from mloda.testing.feature_groups.data_operations.mixins.input_validation import (
    InputValidationCase,
    InputValidationTestMixin,
)
from mloda.testing.feature_groups.data_operations.mixins.output_contract import OutputContractTestMixin

# ---------------------------------------------------------------------------
# Reusable test base class
# ---------------------------------------------------------------------------


class ScalarArithmeticTestBase(InputValidationTestMixin, OutputContractTestMixin, DataOpsTestBase):
    """Abstract base class for scalar arithmetic framework tests."""

    # -- OutputContractTestMixin configuration ----------------------------------

    def output_contract_feature_set(self) -> FeatureSet:
        return make_feature_set("value_int__add_constant", constant=5)

    ALL_OPS = {"add", "subtract", "multiply", "divide"}

    # -- InputValidationTestMixin configuration ---------------------------------

    @classmethod
    def input_validation_cases(cls) -> dict[str, Any]:
        return {
            "multi_column_in_features": InputValidationCase(
                "bad_multi",
                {"arithmetic_op": "add", "in_features": ["value_int", "value_float"], "constant": 5},
                "at most 1",
            ),
            "missing_source_column": InputValidationCase(
                "value_int__add_constant",
                {"constant": 5},
                "Source column 'value_int' is not present",
                data=typed_data_without(DataOperationsTestDataCreator.typed_data(), "value_int"),
            ),
            "empty_partition_by": None,
            "missing_partition_by_column": None,
            "missing_order_by_column": None,
        }

    @classmethod
    def supported_ops(cls) -> set[str]:
        """Subtypes the Implementation declares; the full set for a stub."""
        return cls.declared_supported(cls.ALL_OPS)

    @classmethod
    def reference_calculate(cls, table: pa.Table, features: Any) -> Any:
        """Run the pure Python reference function on the canonical table."""
        from mloda.testing.feature_groups.data_operations.row_preserving.arithmetic.reference import (
            reference_scalar_arithmetic,
        )

        return reference_scalar_arithmetic(cls.feature_group_class(), table, features)

    # -- Concrete tests -----------------------------------------------------

    def test_divide_by_zero_raises(self) -> None:
        """Dividing by zero must raise ValueError."""
        fs = make_feature_set("value_int__divide_constant", constant=0)
        with pytest.raises(ValueError, match="[Dd]ivide|[Zz]ero|0"):
            calculate_with(self.implementation_class(), self.test_data, fs)

    def test_missing_constant_raises(self) -> None:
        """If Options has no constant key, calculate_with must raise ValueError."""
        feature = Feature("value_int__add_constant", options=Options())
        fs = FeatureSet()
        fs.add(feature)
        with pytest.raises(ValueError, match="constant"):
            calculate_with(self.implementation_class(), self.test_data, fs)

    def test_unsupported_operation_raises(self) -> None:
        """An unrecognized arithmetic operation must raise ValueError."""
        feature = Feature(
            "value_int__bogus_constant",
            options=Options(context={"constant": 5}),
        )
        fs = FeatureSet()
        fs.add(feature)
        with pytest.raises(ValueError, match="[Uu]nsupported|[Cc]ould not extract"):
            calculate_with(self.implementation_class(), self.test_data, fs)

    @pytest.mark.parametrize(("op", "constant"), [("add", 5), ("multiply", 2)])
    def test_option_based_config(self, op: str, constant: int) -> None:
        """Option-based configuration (not string pattern) matches the reference."""
        self._compare_with_reference(
            f"my_custom_{op}", use_approx=True, arithmetic_op=op, in_features="value_int", constant=constant
        )

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

        fs = make_feature_set("name__add_constant", constant=5)
        with pytest.raises(ValueError, match=r"(?i)numeric") as exc_info:
            calculate_with(self.implementation_class(), self.test_data, fs)
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
            calculate_with(self.implementation_class(), self.test_data, fs)
        assert "is_active" in str(exc_info.value)

    def test_boolean_source_column_computed_when_undetected(self) -> None:
        """A backend that cannot detect a boolean source computes it as 0/1."""
        if "boolean" in self.detects_non_numeric_source():
            pytest.skip("Backend rejects boolean source columns")
        flags = self._source.to_python_dict()["is_active"]
        expected = [None if f is None else int(f) + 5 for f in flags]
        fs = make_feature_set("is_active__add_constant", constant=5)
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        result_col = self.extract_column(result, "is_active__add_constant")
        assert_values_with_nulls(result_col, expected, approx=True)

    # -- Cross-framework comparison -----------------------------------------

    @pytest.mark.parametrize(
        ("feature_name", "constant"),
        [
            pytest.param("value_int__add_constant", 5, id="add"),
            pytest.param("value_int__subtract_constant", 10, id="subtract"),
            pytest.param("value_int__multiply_constant", 2, id="multiply"),
            pytest.param("value_int__divide_constant", 2.0, id="divide"),
            # Guards against truncating int division.
            pytest.param("value_int__divide_constant", 3, id="divide_int_constant"),
        ],
    )
    def test_cross_framework(self, feature_name: str, constant: int | float) -> None:
        self._compare_with_reference(feature_name, use_approx=True, constant=constant)
