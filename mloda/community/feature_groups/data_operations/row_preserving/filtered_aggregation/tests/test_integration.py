"""Integration tests for filtered aggregation through mloda's full pipeline.

These tests verify that filtered aggregation feature groups work end-to-end
through mloda's runtime, including plugin discovery, feature resolution,
and the PluginCollector mechanism.

Uses the shared DataOpsIntegrationTestBase from the testing library.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.integration import DataOpsIntegrationTestBase
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.filtered_aggregation.pyarrow_filtered_aggregation import (
    PyArrowFilteredAggregation,
)


class TestFilteredAggregationIntegration(DataOpsIntegrationTestBase):
    """Standard integration tests inherited from the base class."""

    @classmethod
    def feature_group_class(cls) -> type:
        return PyArrowFilteredAggregation

    @classmethod
    def data_creator_class(cls) -> type:
        return PyArrowDataOpsTestDataCreator

    @classmethod
    def compute_framework_class(cls) -> type:
        return PyArrowTable

    @classmethod
    def primary_feature_name(cls) -> str:
        return "value_int__sum_filtered_groupby"

    @classmethod
    def primary_feature_options(cls) -> dict[str, Any]:
        return {"partition_by": ["region"], "filter_column": "category", "filter_value": "X"}

    @classmethod
    def primary_expected_values(cls) -> list[Any]:
        return [10, 10, 10, 10, 60, 60, 60, 60, 15, 15, 15, -10]

    @classmethod
    def secondary_feature_name(cls) -> str:
        return "value_int__avg_filtered_groupby"

    @classmethod
    def secondary_feature_options(cls) -> dict[str, Any]:
        return {"partition_by": ["region"], "filter_column": "category", "filter_value": "X"}

    @classmethod
    def secondary_expected_values(cls) -> list[Any]:
        return [5.0, 5.0, 5.0, 5.0, 60.0, 60.0, 60.0, 60.0, 15.0, 15.0, 15.0, -10.0]

    @classmethod
    def valid_feature_names(cls) -> list[str]:
        return [
            "value_int__sum_filtered_groupby",
            "value_int__avg_filtered_groupby",
            "value_int__count_filtered_groupby",
        ]

    @classmethod
    def invalid_feature_names(cls) -> list[str]:
        return ["value_int", "value_int__sum"]

    @classmethod
    def match_options(cls) -> Options:
        return Options(context={"partition_by": ["region"], "filter_column": "category", "filter_value": "X"})

    @classmethod
    def use_approx(cls) -> bool:
        return True


class TestIntegrationMultipleFeatures:
    """Test multiple filtered aggregation features in a single run_all call."""

    def test_sum_and_avg_together(self) -> None:
        """Request both sum and avg filtered features in one pipeline run."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowFilteredAggregation}
        )

        f_sum = Feature(
            "value_int__sum_filtered_groupby",
            options=Options(context={"partition_by": ["region"], "filter_column": "category", "filter_value": "X"}),
        )
        f_avg = Feature(
            "value_int__avg_filtered_groupby",
            options=Options(context={"partition_by": ["region"], "filter_column": "category", "filter_value": "X"}),
        )

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
            if "value_int__sum_filtered_groupby" in table.column_names:
                sum_col = table.column("value_int__sum_filtered_groupby").to_pylist()
                expected_sum = [10, 10, 10, 10, 60, 60, 60, 60, 15, 15, 15, -10]
                assert sum_col == expected_sum
                sum_found = True
            if "value_int__avg_filtered_groupby" in table.column_names:
                avg_col = table.column("value_int__avg_filtered_groupby").to_pylist()
                expected_avg = [5.0, 5.0, 5.0, 5.0, 60.0, 60.0, 60.0, 60.0, 15.0, 15.0, 15.0, -10.0]
                assert avg_col == pytest.approx(expected_avg, rel=1e-3)
                avg_found = True

        assert sum_found, "sum_filtered_groupby result not found in any result table"
        assert avg_found, "avg_filtered_groupby result not found in any result table"

    def test_different_aggregation_types(self) -> None:
        """Request min and max filtered features in one pipeline run."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowFilteredAggregation}
        )

        f_min = Feature(
            "value_int__min_filtered_groupby",
            options=Options(context={"partition_by": ["region"], "filter_column": "category", "filter_value": "X"}),
        )
        f_max = Feature(
            "value_int__max_filtered_groupby",
            options=Options(context={"partition_by": ["region"], "filter_column": "category", "filter_value": "X"}),
        )

        results = mloda.run_all(
            [f_min, f_max],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        min_found = False
        max_found = False
        for table in results:
            if not isinstance(table, pa.Table):
                continue
            if "value_int__min_filtered_groupby" in table.column_names:
                min_col = table.column("value_int__min_filtered_groupby").to_pylist()
                expected_min = [0, 0, 0, 0, 60, 60, 60, 60, 15, 15, 15, -10]
                assert min_col == expected_min
                min_found = True
            if "value_int__max_filtered_groupby" in table.column_names:
                max_col = table.column("value_int__max_filtered_groupby").to_pylist()
                expected_max = [10, 10, 10, 10, 60, 60, 60, 60, 15, 15, 15, -10]
                assert max_col == expected_max
                max_found = True

        assert min_found, "min_filtered_groupby result not found in any result table"
        assert max_found, "max_filtered_groupby result not found in any result table"
