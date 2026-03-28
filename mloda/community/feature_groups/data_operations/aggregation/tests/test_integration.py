"""Integration tests for column aggregation through mloda's full pipeline.

Covers both string-pattern features (e.g. ``value_int__sum_aggr``) and
option-based configuration (aggregation_type + in_features via Options).
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.aggregation import (
    EXPECTED_AVG,
    EXPECTED_MAX,
    EXPECTED_MIN,
    EXPECTED_SUM,
)
from mloda.testing.feature_groups.data_operations.integration import DataOpsIntegrationTestBase
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.aggregation.pyarrow_aggregation import (
    PyArrowColumnAggregation,
)


class TestAggregationIntegration(DataOpsIntegrationTestBase):
    @classmethod
    def feature_group_class(cls) -> type:
        return PyArrowColumnAggregation

    @classmethod
    def data_creator_class(cls) -> type:
        return PyArrowDataOpsTestDataCreator

    @classmethod
    def compute_framework_class(cls) -> type:
        return PyArrowTable

    @classmethod
    def primary_feature_name(cls) -> str:
        return "value_int__sum_aggr"

    @classmethod
    def primary_feature_options(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def primary_expected_values(cls) -> list[Any]:
        return [EXPECTED_SUM] * 12

    @classmethod
    def secondary_feature_name(cls) -> str:
        return "value_int__avg_aggr"

    @classmethod
    def secondary_feature_options(cls) -> dict[str, Any]:
        return {}

    @classmethod
    def secondary_expected_values(cls) -> list[Any]:
        return [EXPECTED_AVG] * 12

    @classmethod
    def valid_feature_names(cls) -> list[str]:
        return ["value_int__sum_aggr", "value_int__avg_aggr", "value_int__count_aggr"]

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
            {PyArrowDataOpsTestDataCreator, PyArrowColumnAggregation}
        )

        f_sum = Feature("value_int__sum_aggr", options=Options())
        f_avg = Feature("value_int__avg_aggr", options=Options())

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
            if "value_int__sum_aggr" in table.column_names:
                sum_col = table.column("value_int__sum_aggr").to_pylist()
                assert all(v == EXPECTED_SUM for v in sum_col)
                sum_found = True
            if "value_int__avg_aggr" in table.column_names:
                avg_col = table.column("value_int__avg_aggr").to_pylist()
                assert all(v == pytest.approx(EXPECTED_AVG, rel=1e-6) for v in avg_col)
                avg_found = True

        assert sum_found, "sum_aggr result not found"
        assert avg_found, "avg_aggr result not found"


class TestIntegrationOptionBasedConfig:
    """Integration tests for option-based (non-string-pattern) configuration.

    These tests exercise the path where aggregation_type and in_features
    are provided through Options rather than encoded in the feature name.
    """

    def test_option_based_sum_through_pipeline(self) -> None:
        """Option-based sum produces the same result as the string pattern."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowColumnAggregation}
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
            {PyArrowDataOpsTestDataCreator, PyArrowColumnAggregation}
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
            {PyArrowDataOpsTestDataCreator, PyArrowColumnAggregation}
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
            {PyArrowDataOpsTestDataCreator, PyArrowColumnAggregation}
        )

        f_pattern = Feature("value_int__sum_aggr", options=Options())
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
            if "value_int__sum_aggr" in table.column_names:
                col = table.column("value_int__sum_aggr").to_pylist()
                assert all(v == EXPECTED_SUM for v in col)
                sum_found = True
            if "my_avg" in table.column_names:
                col = table.column("my_avg").to_pylist()
                assert all(v == pytest.approx(EXPECTED_AVG, rel=1e-6) for v in col)
                avg_found = True

        assert sum_found, "string-pattern sum result not found"
        assert avg_found, "option-based avg result not found"
