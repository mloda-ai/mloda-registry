"""Integration tests for point arithmetic through mloda's full pipeline.

Covers both string-pattern features (e.g. ``value_int&amount__add_point``) and
option-based configuration (arithmetic_op + in_features via Options).
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.row_preserving.point_arithmetic.point_arithmetic import (
    EXPECTED_ADD,
    EXPECTED_MULTIPLY,
    EXPECTED_SUBTRACT,
)
from mloda.testing.feature_groups.data_operations.integration import DataOpsIntegrationTestBase
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.pyarrow_point_arithmetic import (
    PyArrowPointArithmetic,
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


class TestPointArithmeticIntegration(DataOpsIntegrationTestBase):
    @classmethod
    def feature_group_class(cls) -> type:
        return PyArrowPointArithmetic

    @classmethod
    def data_creator_class(cls) -> type:
        return PyArrowDataOpsTestDataCreator

    @classmethod
    def compute_framework_class(cls) -> type:
        return PyArrowTable

    @classmethod
    def primary_feature_name(cls) -> str:
        return "value_int&amount__add_point"

    @classmethod
    def primary_feature_options(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def primary_expected_values(cls) -> list[Any]:
        return list(EXPECTED_ADD)

    @classmethod
    def secondary_feature_name(cls) -> str:
        return "value_int&amount__multiply_point"

    @classmethod
    def secondary_feature_options(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def secondary_expected_values(cls) -> list[Any]:
        return list(EXPECTED_MULTIPLY)

    @classmethod
    def valid_feature_names(cls) -> list[str]:
        return [
            "value_int&amount__add_point",
            "value_int&amount__subtract_point",
            "value_int&amount__multiply_point",
            "value_int&amount__divide_point",
        ]

    @classmethod
    def invalid_feature_names(cls) -> list[str]:
        return [
            "value_int",
            "value_int__add",
            # This name matches scalar_arithmetic, NOT point_arithmetic.
            "value_int&amount__add_constant",
            # This is a scalar_arithmetic-style name (no & separator), should not match point.
            "value_int__add_amount",
        ]

    @classmethod
    def match_options(cls) -> Options:
        return Options()

    @classmethod
    def use_approx(cls) -> bool:
        return True

    def test_match_rejects_missing_config(self) -> None:
        """Override: point_arithmetic pattern features match even without any Options.

        The string pattern carries the operation and both source columns;
        no context is required for pattern-based matching.
        """
        options = Options()
        fg_cls = self.feature_group_class()
        assert fg_cls.match_feature_group_criteria(self.primary_feature_name(), options)  # type: ignore[attr-defined]


class TestIntegrationMultipleFeatures:
    def test_add_and_multiply_together(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowPointArithmetic}
        )

        f_add = Feature("value_int&amount__add_point")
        f_mul = Feature("value_int&amount__multiply_point")

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
            if "value_int&amount__add_point" in table.column_names:
                col = table.column("value_int&amount__add_point").to_pylist()
                assert _approx_equal(col, EXPECTED_ADD)
                add_found = True
            if "value_int&amount__multiply_point" in table.column_names:
                col = table.column("value_int&amount__multiply_point").to_pylist()
                assert _approx_equal(col, EXPECTED_MULTIPLY)
                mul_found = True

        assert add_found, "add_point result not found"
        assert mul_found, "multiply_point result not found"


class TestIntegrationOptionBasedConfig:
    """Integration tests for option-based (non-string-pattern) configuration."""

    def test_option_based_add_through_pipeline(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowPointArithmetic}
        )

        feature = Feature(
            "my_custom_add",
            options=Options(
                context={
                    "arithmetic_op": "add",
                    "in_features": ["value_int", "amount"],
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
                assert _approx_equal(col, EXPECTED_ADD)
                assert len(col) == 12
                found = True

        assert found, "option-based add result not found"

    def test_option_based_subtract_through_pipeline(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowPointArithmetic}
        )

        feature = Feature(
            "my_diff",
            options=Options(
                context={
                    "arithmetic_op": "subtract",
                    "in_features": ["value_int", "amount"],
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
            if "my_diff" in table.column_names:
                col = table.column("my_diff").to_pylist()
                assert _approx_equal(col, EXPECTED_SUBTRACT)
                found = True

        assert found, "option-based subtract result not found"

    def test_string_and_option_features_together(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowPointArithmetic}
        )

        f_pattern = Feature("value_int&amount__add_point")
        f_option = Feature(
            "my_mul",
            options=Options(
                context={
                    "arithmetic_op": "multiply",
                    "in_features": ["value_int", "amount"],
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
            if "value_int&amount__add_point" in table.column_names:
                col = table.column("value_int&amount__add_point").to_pylist()
                assert _approx_equal(col, EXPECTED_ADD)
                add_found = True
            if "my_mul" in table.column_names:
                col = table.column("my_mul").to_pylist()
                assert _approx_equal(col, EXPECTED_MULTIPLY)
                mul_found = True

        assert add_found, "string-pattern add result not found"
        assert mul_found, "option-based multiply result not found"
