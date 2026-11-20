"""Shared test base for two-column point arithmetic; expected values come from the reference."""

from __future__ import annotations

import math
import re
from typing import Any

import pyarrow as pa
import pytest
from mloda.provider import FeatureSet
from mloda.user import Feature, Options

from mloda.testing.data_creator.base import DataOperationsTestDataCreator
from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase, divergent_subtypes
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

# Source column used by the divide-by-integer regression guard
VALUE_INT: list[int | None] = [10, -5, 0, 20, None, 50, 30, 60, 15, 15, 40, -10]


# ---------------------------------------------------------------------------
# Reusable test base class
# ---------------------------------------------------------------------------


class PointArithmeticTestBase(InputValidationTestMixin, OutputContractTestMixin, DataOpsTestBase):
    """Abstract base class for two-column point arithmetic framework tests."""

    # -- OutputContractTestMixin configuration ----------------------------------

    def output_contract_feature_set(self) -> FeatureSet:
        return make_feature_set("value_int&amount__add_point")

    ALL_OPS = {"add", "subtract", "multiply", "divide"}

    # -- InputValidationTestMixin configuration ---------------------------------

    @classmethod
    def input_validation_cases(cls) -> dict[str, Any]:
        return {
            "multi_column_in_features": InputValidationCase(
                "bad_too_many",
                {"arithmetic_op": "add", "in_features": ["value_int", "amount", "value_float"]},
                "at most 2",
            ),
            "missing_source_column": InputValidationCase(
                "value_int&amount__add_point",
                {},
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
            reference_point_arithmetic,
        )

        return reference_point_arithmetic(cls.feature_group_class(), table, features)

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
        """Whether divide-by-zero produces IEEE-754 inf/nan, read from the Implementation's ``divergences``.

        An Implementation that names ``divide`` in a divergence record (SQLite: no IEEE-754
        storage, NULL for divide-by-zero rows) does not propagate inf.
        """
        return "divide" not in divergent_subtypes(cls.implementation_class())

    # -- Concrete tests -----------------------------------------------------

    def test_unsupported_operation_raises(self) -> None:
        """An unrecognized arithmetic operation must raise ValueError."""
        feature = Feature(
            "value_int&amount__bogus_point",
            options=Options(),
        )
        fs = FeatureSet()
        fs.add(feature)
        with pytest.raises(ValueError, match="[Uu]nsupported|[Cc]ould not extract"):
            calculate_with(self.implementation_class(), self.test_data, fs)

    @pytest.mark.parametrize("op", ["add", "subtract", "multiply"])
    def test_option_based_config(self, op: str) -> None:
        """Option-based configuration (not string pattern) matches the reference."""
        self._compare_with_reference(
            f"my_custom_{op}", use_approx=True, arithmetic_op=op, in_features=["value_int", "amount"]
        )

    # -- In-feature count enforcement at compute time -----------------------

    def test_too_few_in_features_rejected_at_calculate(self) -> None:
        """calculate_with must reject features with fewer than 2 in_features."""
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
            calculate_with(self.implementation_class(), self.test_data, fs)

    # -- Source-column dtype enforcement ------------------------------------

    @pytest.mark.parametrize(
        "feature_name",
        [
            pytest.param("name&value_int__add_point", id="col_a"),
            pytest.param("value_int&name__add_point", id="col_b"),
        ],
    )
    def test_string_source_column_rejected(self, feature_name: str) -> None:
        """A non-numeric (string) source column must raise ValueError naming the column."""
        if "string" not in self.detects_non_numeric_source():
            pytest.skip("Backend cannot distinguish string source columns at relation level")

        fs = make_feature_set(feature_name)
        with pytest.raises(ValueError, match=r"(?i)numeric") as exc_info:
            calculate_with(self.implementation_class(), self.test_data, fs)
        assert re.search(r"['\"]name['\"]", str(exc_info.value)), (
            f"Expected source column 'name' to be named (quoted) in the error message, got: {exc_info.value!r}"
        )

    @pytest.mark.parametrize(
        "feature_name",
        [
            pytest.param("is_active&value_int__add_point", id="col_a"),
            pytest.param("value_int&is_active__add_point", id="col_b"),
        ],
    )
    def test_boolean_source_column_rejected(self, feature_name: str) -> None:
        """A boolean source column must raise ValueError naming the column."""
        if "boolean" not in self.detects_non_numeric_source():
            pytest.skip("Backend cannot distinguish boolean source columns at relation level")
        fs = make_feature_set(feature_name)
        with pytest.raises(ValueError, match=r"(?i)numeric") as exc_info:
            calculate_with(self.implementation_class(), self.test_data, fs)
        assert "is_active" in str(exc_info.value)

    def test_boolean_source_column_computed_when_undetected(self) -> None:
        """A backend that cannot detect a boolean source computes it as 0/1 in either position."""
        if "boolean" in self.detects_non_numeric_source():
            pytest.skip("Backend rejects boolean source columns")
        values = self._source.to_python_dict()["value_int"]
        flags = self._source.to_python_dict()["is_active"]
        expected = [None if v is None or f is None else v + int(f) for v, f in zip(values, flags)]
        for name in ("is_active&value_int__add_point", "value_int&is_active__add_point"):
            result = calculate_with(self.implementation_class(), self.test_data, make_feature_set(name))
            result_col = self.extract_column(result, name)
            assert_values_with_nulls(result_col, expected, approx=True)

    # -- Cross-framework comparison -----------------------------------------

    @pytest.mark.parametrize("op", ["add", "subtract", "multiply", "divide"])
    def test_cross_framework(self, op: str) -> None:
        """Compare against the reference; divide-by-zero rows are None where the backend declares the divergence."""
        feature_name = f"value_int&amount__{op}_point"
        if op != "divide" or self.divide_zero_propagates_inf():
            self._compare_with_reference(feature_name, use_approx=True)
            return
        fs = make_feature_set(feature_name)
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        ref = self.reference_calculate(self._arrow_table, fs)

        actual = self.extract_column(result, feature_name)
        expected = [
            None if isinstance(v, float) and not math.isfinite(v) else v for v in ref.column(feature_name).to_pylist()
        ]
        assert_values_with_nulls(actual, expected, approx=True)

    def test_divide_by_integer_constant_returns_float_per_row(self) -> None:
        """Regression guard: int/int divide must still produce float results."""
        fs = make_feature_set("value_int&value_int__divide_point")
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        result_col = self.extract_column(result, "value_int&value_int__divide_point")

        for i, value in enumerate(result_col):
            if VALUE_INT[i] is None or VALUE_INT[i] == 0:
                continue
            assert isinstance(value, float), f"row {i}: expected float got {type(value).__name__} ({value!r})"
