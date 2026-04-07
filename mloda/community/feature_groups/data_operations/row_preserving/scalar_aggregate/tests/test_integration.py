"""Integration tests for scalar aggregate through mloda's full pipeline.

Covers both string-pattern features (e.g. ``value_int__sum_scalar``) and
option-based configuration (aggregation_type + in_features via Options).
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.row_preserving.scalar_aggregate.scalar_aggregate import (
    EXPECTED_AVG,
    EXPECTED_MAX,
    EXPECTED_MIN,
    EXPECTED_SUM,
)
from mloda.testing.feature_groups.data_operations.integration import DataOpsIntegrationTestBase
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.pyarrow_scalar_aggregate import (
    PyArrowScalarAggregate,
)


class TestScalarAggregateIntegration(DataOpsIntegrationTestBase):
    @classmethod
    def feature_group_class(cls) -> type:
        return PyArrowScalarAggregate

    @classmethod
    def data_creator_class(cls) -> type:
        return PyArrowDataOpsTestDataCreator

    @classmethod
    def compute_framework_class(cls) -> type:
        return PyArrowTable

    @classmethod
    def primary_feature_name(cls) -> str:
        return "value_int__sum_scalar"

    @classmethod
    def primary_feature_options(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def primary_expected_values(cls) -> list[Any]:
        return [EXPECTED_SUM] * 12

    @classmethod
    def secondary_feature_name(cls) -> str:
        return "value_int__avg_scalar"

    @classmethod
    def secondary_feature_options(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def secondary_expected_values(cls) -> list[Any]:
        return [EXPECTED_AVG] * 12

    @classmethod
    def valid_feature_names(cls) -> list[str]:
        return ["value_int__sum_scalar", "value_int__avg_scalar", "value_int__count_scalar"]

    @classmethod
    def invalid_feature_names(cls) -> list[str]:
        return ["value_int", "value_int__sum"]

    @classmethod
    def match_options(cls) -> Options:
        return Options()

    @classmethod
    def use_approx(cls) -> bool:
        return True

    def test_match_rejects_missing_config(self) -> None:
        """Override: aggregation requires no config for pattern-based features."""
        options = Options()
        fg_cls = self.feature_group_class()
        assert fg_cls.match_feature_group_criteria(self.primary_feature_name(), options)  # type: ignore[attr-defined]


class TestIntegrationMultipleFeatures:
    def test_sum_and_avg_together(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowScalarAggregate}
        )

        f_sum = Feature("value_int__sum_scalar", options=Options())
        f_avg = Feature("value_int__avg_scalar", options=Options())

        results = mloda.run_all(
            [f_sum, f_avg],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        sum_found = False
        avg_found = False
        for table in results:
            if not isinstance(table, pa.Table):
                continue
            if "value_int__sum_scalar" in table.column_names:
                sum_col = table.column("value_int__sum_scalar").to_pylist()
                assert all(v == EXPECTED_SUM for v in sum_col)
                sum_found = True
            if "value_int__avg_scalar" in table.column_names:
                avg_col = table.column("value_int__avg_scalar").to_pylist()
                assert all(v == pytest.approx(EXPECTED_AVG, rel=1e-6) for v in avg_col)
                avg_found = True

        assert sum_found, "sum_scalar result not found"
        assert avg_found, "avg_scalar result not found"


class TestIntegrationOptionBasedConfig:
    """Integration tests for option-based (non-string-pattern) configuration.

    These tests exercise the path where aggregation_type and in_features
    are provided through Options rather than encoded in the feature name.
    """

    def test_option_based_sum_through_pipeline(self) -> None:
        """Option-based sum produces the same result as the string pattern."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowScalarAggregate}
        )

        feature = Feature(
            "my_custom_sum",
            options=Options(
                context={
                    "aggregation_type": "sum",
                    "in_features": "value_int",
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
            if "my_custom_sum" in table.column_names:
                col = table.column("my_custom_sum").to_pylist()
                assert all(v == EXPECTED_SUM for v in col)
                assert len(col) == 12
                found = True

        assert found, "option-based sum result not found"

    def test_option_based_min_through_pipeline(self) -> None:
        """Option-based min aggregation through the full pipeline."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowScalarAggregate}
        )

        feature = Feature(
            "lowest_value",
            options=Options(
                context={
                    "aggregation_type": "min",
                    "in_features": "value_int",
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
            if "lowest_value" in table.column_names:
                col = table.column("lowest_value").to_pylist()
                assert all(v == EXPECTED_MIN for v in col)
                found = True

        assert found, "option-based min result not found"

    def test_option_based_max_through_pipeline(self) -> None:
        """Option-based max aggregation through the full pipeline."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowScalarAggregate}
        )

        feature = Feature(
            "highest_value",
            options=Options(
                context={
                    "aggregation_type": "max",
                    "in_features": "value_int",
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
            if "highest_value" in table.column_names:
                col = table.column("highest_value").to_pylist()
                assert all(v == EXPECTED_MAX for v in col)
                found = True

        assert found, "option-based max result not found"

    def test_string_and_option_features_together(self) -> None:
        """Combining string-pattern and option-based features in one run."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowScalarAggregate}
        )

        f_pattern = Feature("value_int__sum_scalar", options=Options())
        f_option = Feature(
            "my_avg",
            options=Options(
                context={
                    "aggregation_type": "avg",
                    "in_features": "value_int",
                }
            ),
        )

        results = mloda.run_all(
            [f_pattern, f_option],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        sum_found = False
        avg_found = False
        for table in results:
            if not isinstance(table, pa.Table):
                continue
            if "value_int__sum_scalar" in table.column_names:
                col = table.column("value_int__sum_scalar").to_pylist()
                assert all(v == EXPECTED_SUM for v in col)
                sum_found = True
            if "my_avg" in table.column_names:
                col = table.column("my_avg").to_pylist()
                assert all(v == pytest.approx(EXPECTED_AVG, rel=1e-6) for v in col)
                avg_found = True

        assert sum_found, "string-pattern sum result not found"
        assert avg_found, "option-based avg result not found"


def _extract_result_column(results: list[Any], feature_name: str) -> list[Any]:
    for table in results:
        if isinstance(table, pa.Table) and feature_name in table.column_names:
            result: list[Any] = table.column(feature_name).to_pylist()
            return result
    raise AssertionError(f"No result table with {feature_name} found")


class TestScalarAggregateMaskIntegration:
    """Integration tests for scalar aggregate with conditional mask."""

    def test_mask_single_condition(self) -> None:
        """Masked sum_scalar through full pipeline: only category='X' rows contribute."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowScalarAggregate}
        )
        feature = Feature(
            "value_int__sum_scalar",
            options=Options(context={"mask": ("category", "equal", "X")}),
        )
        results = mloda.run_all([feature], compute_frameworks={PyArrowTable}, plugin_collector=plugin_collector)
        result_col = _extract_result_column(results, "value_int__sum_scalar")
        assert len(result_col) == 12
        assert all(v == 75 for v in result_col)

    def test_mask_and_unmasked_produce_different_results(self) -> None:
        """Masked and unmasked scalar features produce different results from the same source."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowScalarAggregate}
        )

        # Masked sum: only category='X' rows contribute
        f_masked = Feature(
            "value_int__sum_scalar",
            options=Options(context={"mask": ("category", "equal", "X")}),
        )
        masked_results = mloda.run_all([f_masked], compute_frameworks={PyArrowTable}, plugin_collector=plugin_collector)
        masked_col = _extract_result_column(masked_results, "value_int__sum_scalar")
        assert len(masked_col) == 12
        assert all(v == 75 for v in masked_col)

        # Unmasked count: all non-null value_int rows contribute
        f_unmasked = Feature(
            "value_int__count_scalar",
            options=Options(context={}),
        )
        unmasked_results = mloda.run_all(
            [f_unmasked], compute_frameworks={PyArrowTable}, plugin_collector=plugin_collector
        )
        unmasked_col = _extract_result_column(unmasked_results, "value_int__count_scalar")
        assert len(unmasked_col) == 12
        assert all(v == 11 for v in unmasked_col)
