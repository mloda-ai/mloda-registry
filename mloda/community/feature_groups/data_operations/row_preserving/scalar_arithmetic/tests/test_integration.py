"""Integration tests for scalar arithmetic through mloda's full pipeline.

Covers both string-pattern features (e.g. ``value_int__add_constant``) and
option-based configuration (arithmetic_op + in_features + constant via Options).
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.row_preserving.scalar_arithmetic.scalar_arithmetic import (
    EXPECTED_ADD_5,
    EXPECTED_DIVIDE_2,
    EXPECTED_MULTIPLY_2,
    EXPECTED_SUBTRACT_10,
)
from mloda.testing.feature_groups.data_operations.integration import DataOpsIntegrationTestBase
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.pyarrow_scalar_arithmetic import (
    PyArrowScalarArithmetic,
)


def _approx_equal(actual: list[Any], expected: list[Any]) -> bool:
    if len(actual) != len(expected):
        return False
    for a, e in zip(actual, expected):
        if e is None:
            if a is not None:
                return False
        else:
            if a is None or a != pytest.approx(e, rel=1e-6):
                return False
    return True


class TestScalarArithmeticIntegration(DataOpsIntegrationTestBase):
    @classmethod
    def feature_group_class(cls) -> type:
        return PyArrowScalarArithmetic

    @classmethod
    def data_creator_class(cls) -> type:
        return PyArrowDataOpsTestDataCreator

    @classmethod
    def compute_framework_class(cls) -> type:
        return PyArrowTable

    @classmethod
    def primary_feature_name(cls) -> str:
        return "value_int__add_constant"

    @classmethod
    def primary_feature_options(cls) -> dict[str, Any]:
        return {"constant": 5}

    @classmethod
    def primary_expected_values(cls) -> list[Any]:
        return list(EXPECTED_ADD_5)

    @classmethod
    def secondary_feature_name(cls) -> str:
        return "value_int__multiply_constant"

    @classmethod
    def secondary_feature_options(cls) -> dict[str, Any]:
        return {"constant": 2}

    @classmethod
    def secondary_expected_values(cls) -> list[Any]:
        return list(EXPECTED_MULTIPLY_2)

    @classmethod
    def valid_feature_names(cls) -> list[str]:
        return [
            "value_int__add_constant",
            "value_int__subtract_constant",
            "value_int__multiply_constant",
            "value_int__divide_constant",
        ]

    @classmethod
    def invalid_feature_names(cls) -> list[str]:
        return ["value_int", "value_int__add", "value_int__add_scalar"]

    @classmethod
    def match_options(cls) -> Options:
        # CONSTANT has strict_validation=False; pattern-based match succeeds without it.
        return Options()

    @classmethod
    def use_approx(cls) -> bool:
        return True

    def test_match_rejects_missing_config(self) -> None:
        """Override: arithmetic pattern features match even without constant in Options.

        Match validation is permissive on CONSTANT (strict_validation=False);
        the ValueError comes at compute time when the constant is missing.
        """
        options = Options()
        fg_cls = self.feature_group_class()
        assert fg_cls.match_feature_group_criteria(self.primary_feature_name(), options)  # type: ignore[attr-defined]


class TestIntegrationMultipleFeatures:
    def test_add_and_multiply_together(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowScalarArithmetic}
        )

        f_add = Feature("value_int__add_constant", options=Options(context={"constant": 5}))
        f_mul = Feature("value_int__multiply_constant", options=Options(context={"constant": 2}))

        results = mloda.run_all(
            [f_add, f_mul],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        add_found = False
        mul_found = False
        for table in results:
            if not isinstance(table, pa.Table):
                continue
            if "value_int__add_constant" in table.column_names:
                col = table.column("value_int__add_constant").to_pylist()
                assert _approx_equal(col, EXPECTED_ADD_5)
                add_found = True
            if "value_int__multiply_constant" in table.column_names:
                col = table.column("value_int__multiply_constant").to_pylist()
                assert _approx_equal(col, EXPECTED_MULTIPLY_2)
                mul_found = True

        assert add_found, "add_constant result not found"
        assert mul_found, "multiply_constant result not found"


class TestIntegrationOptionBasedConfig:
    """Integration tests for option-based (non-string-pattern) configuration."""

    def test_option_based_add_through_pipeline(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowScalarArithmetic}
        )

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

        results = mloda.run_all(
            [feature],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        found = False
        for table in results:
            if not isinstance(table, pa.Table):
                continue
            if "my_custom_add" in table.column_names:
                col = table.column("my_custom_add").to_pylist()
                assert _approx_equal(col, EXPECTED_ADD_5)
                assert len(col) == 12
                found = True

        assert found, "option-based add result not found"

    def test_option_based_subtract_through_pipeline(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowScalarArithmetic}
        )

        feature = Feature(
            "minus_ten",
            options=Options(
                context={
                    "arithmetic_op": "subtract",
                    "in_features": "value_int",
                    "constant": 10,
                }
            ),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        found = False
        for table in results:
            if not isinstance(table, pa.Table):
                continue
            if "minus_ten" in table.column_names:
                col = table.column("minus_ten").to_pylist()
                assert _approx_equal(col, EXPECTED_SUBTRACT_10)
                found = True

        assert found, "option-based subtract result not found"

    def test_option_based_divide_through_pipeline(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowScalarArithmetic}
        )

        feature = Feature(
            "halved",
            options=Options(
                context={
                    "arithmetic_op": "divide",
                    "in_features": "value_int",
                    "constant": 2.0,
                }
            ),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        found = False
        for table in results:
            if not isinstance(table, pa.Table):
                continue
            if "halved" in table.column_names:
                col = table.column("halved").to_pylist()
                assert _approx_equal(col, EXPECTED_DIVIDE_2)
                found = True

        assert found, "option-based divide result not found"

    def test_string_and_option_features_together(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowScalarArithmetic}
        )

        f_pattern = Feature(
            "value_int__add_constant",
            options=Options(context={"constant": 5}),
        )
        f_option = Feature(
            "my_mul",
            options=Options(
                context={
                    "arithmetic_op": "multiply",
                    "in_features": "value_int",
                    "constant": 2,
                }
            ),
        )

        results = mloda.run_all(
            [f_pattern, f_option],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        add_found = False
        mul_found = False
        for table in results:
            if not isinstance(table, pa.Table):
                continue
            if "value_int__add_constant" in table.column_names:
                col = table.column("value_int__add_constant").to_pylist()
                assert _approx_equal(col, EXPECTED_ADD_5)
                add_found = True
            if "my_mul" in table.column_names:
                col = table.column("my_mul").to_pylist()
                assert _approx_equal(col, EXPECTED_MULTIPLY_2)
                mul_found = True

        assert add_found, "string-pattern add result not found"
        assert mul_found, "option-based multiply result not found"
