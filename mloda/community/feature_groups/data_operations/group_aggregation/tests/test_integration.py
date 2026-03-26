"""Integration tests for group aggregation through mloda's full pipeline.

These tests verify that group aggregation feature groups work end-to-end
through mloda's runtime, including plugin discovery, feature resolution,
and the PluginCollector mechanism.

Uses the shared DataOpsIntegrationTestBase from the testing library,
plus custom multi-feature tests for group aggregation specifics.

Note: mloda's pipeline only returns requested feature columns, so
integration tests verify values as sets (not keyed by partition columns).
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

from mloda.community.feature_groups.data_operations.group_aggregation.pyarrow_group_aggregation import (
    PyArrowGroupAggregation,
)


class TestGroupAggregationIntegration(DataOpsIntegrationTestBase):
    """Standard integration tests inherited from the base class."""

    @classmethod
    def feature_group_class(cls) -> type:
        return PyArrowGroupAggregation

    @classmethod
    def data_creator_class(cls) -> type:
        return PyArrowDataOpsTestDataCreator

    @classmethod
    def compute_framework_class(cls) -> type:
        return PyArrowTable

    @classmethod
    def primary_feature_name(cls) -> str:
        return "value_int__sum_grouped"

    @classmethod
    def primary_feature_options(cls) -> dict[str, Any]:
        return {"partition_by": ["region"]}

    @classmethod
    def primary_expected_values(cls) -> list[Any]:
        return [25, 140, 70, -10]

    @classmethod
    def secondary_feature_name(cls) -> str:
        return "value_int__avg_grouped"

    @classmethod
    def secondary_feature_options(cls) -> dict[str, Any]:
        return {"partition_by": ["region"]}

    @classmethod
    def secondary_expected_values(cls) -> list[Any]:
        return [6.25, 46.667, 23.333, -10.0]

    @classmethod
    def valid_feature_names(cls) -> list[str]:
        return ["value_int__sum_grouped", "value_int__avg_grouped", "value_int__count_grouped"]

    @classmethod
    def invalid_feature_names(cls) -> list[str]:
        return ["value_int", "value_int__sum"]

    @classmethod
    def match_options(cls) -> Options:
        return Options(context={"partition_by": ["region"]})

    @classmethod
    def expected_row_count(cls) -> int:
        return 4

    @classmethod
    def compare_sorted(cls) -> bool:
        return True

    @classmethod
    def use_approx(cls) -> bool:
        return True


class TestIntegrationAdditionalAggTypes:
    """Test additional aggregation types through the full mloda pipeline.

    Group aggregation reduces rows, so each feature must be requested
    in a separate run_all call (mloda cannot merge different row-count
    outputs without explicit Links).
    """

    def test_min_grouped_through_pipeline(self) -> None:
        """Run value_int__min_grouped through run_all and verify the result."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowGroupAggregation}
        )

        feature = Feature(
            "value_int__min_grouped",
            options=Options(context={"partition_by": ["region"]}),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        result_table = None
        for table in results:
            if isinstance(table, pa.Table) and "value_int__min_grouped" in table.column_names:
                result_table = table
                break

        assert result_table is not None
        assert result_table.num_rows == 4
        min_col = sorted(result_table.column("value_int__min_grouped").to_pylist())
        assert min_col == sorted([-5, 30, 15, -10])

    def test_max_grouped_through_pipeline(self) -> None:
        """Run value_int__max_grouped through run_all and verify the result."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowGroupAggregation}
        )

        feature = Feature(
            "value_int__max_grouped",
            options=Options(context={"partition_by": ["region"]}),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        result_table = None
        for table in results:
            if isinstance(table, pa.Table) and "value_int__max_grouped" in table.column_names:
                result_table = table
                break

        assert result_table is not None
        assert result_table.num_rows == 4
        max_col = sorted(result_table.column("value_int__max_grouped").to_pylist())
        assert max_col == sorted([20, 60, 40, -10])

    def test_std_grouped_through_pipeline(self) -> None:
        """Run value_int__std_grouped through run_all and verify the result."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowGroupAggregation}
        )

        feature = Feature(
            "value_int__std_grouped",
            options=Options(context={"partition_by": ["region"]}),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        result_table = None
        for table in results:
            if isinstance(table, pa.Table) and "value_int__std_grouped" in table.column_names:
                result_table = table
                break

        assert result_table is not None
        assert result_table.num_rows == 4

    def test_median_grouped_through_pipeline(self) -> None:
        """Run value_int__median_grouped through run_all and verify the result."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowGroupAggregation}
        )

        feature = Feature(
            "value_int__median_grouped",
            options=Options(context={"partition_by": ["region"]}),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        result_table = None
        for table in results:
            if isinstance(table, pa.Table) and "value_int__median_grouped" in table.column_names:
                result_table = table
                break

        assert result_table is not None
        assert result_table.num_rows == 4

    def test_nunique_grouped_through_pipeline(self) -> None:
        """Run value_int__nunique_grouped through run_all and verify the result."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowGroupAggregation}
        )

        feature = Feature(
            "value_int__nunique_grouped",
            options=Options(context={"partition_by": ["region"]}),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        result_table = None
        for table in results:
            if isinstance(table, pa.Table) and "value_int__nunique_grouped" in table.column_names:
                result_table = table
                break

        assert result_table is not None
        assert result_table.num_rows == 4


class TestIntegrationMultipleFeatures:
    """Test multiple group aggregation features in separate pipeline runs.

    Group aggregation reduces rows, so each feature must be requested
    in a separate run_all call (mloda cannot merge different row-count
    outputs without explicit Links). These tests verify that different
    aggregation types produce consistent, correct results from the
    same source data.
    """

    def test_sum_and_avg_consistent(self) -> None:
        """Run sum and avg separately and verify both produce correct results."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowGroupAggregation}
        )

        f_sum = Feature(
            "value_int__sum_grouped",
            options=Options(context={"partition_by": ["region"]}),
        )
        sum_results = mloda.run_all(
            [f_sum],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        sum_table = None
        for table in sum_results:
            if isinstance(table, pa.Table) and "value_int__sum_grouped" in table.column_names:
                sum_table = table
                break
        assert sum_table is not None
        sum_col = sorted(sum_table.column("value_int__sum_grouped").to_pylist())
        assert sum_col == sorted([25, 140, 70, -10])

        f_avg = Feature(
            "value_int__avg_grouped",
            options=Options(context={"partition_by": ["region"]}),
        )
        avg_results = mloda.run_all(
            [f_avg],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        avg_table = None
        for table in avg_results:
            if isinstance(table, pa.Table) and "value_int__avg_grouped" in table.column_names:
                avg_table = table
                break
        assert avg_table is not None
        avg_col = sorted(avg_table.column("value_int__avg_grouped").to_pylist())
        expected_avg = sorted([6.25, 46.667, 23.333, -10.0])
        assert avg_col == pytest.approx(expected_avg, rel=1e-3)

        assert sum_table.num_rows == avg_table.num_rows == 4

    def test_min_and_max_consistent(self) -> None:
        """Run min and max separately and verify both produce correct results."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowGroupAggregation}
        )

        f_min = Feature(
            "value_int__min_grouped",
            options=Options(context={"partition_by": ["region"]}),
        )
        min_results = mloda.run_all(
            [f_min],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        min_table = None
        for table in min_results:
            if isinstance(table, pa.Table) and "value_int__min_grouped" in table.column_names:
                min_table = table
                break
        assert min_table is not None
        min_col = sorted(min_table.column("value_int__min_grouped").to_pylist())
        assert min_col == sorted([-5, 30, 15, -10])

        f_max = Feature(
            "value_int__max_grouped",
            options=Options(context={"partition_by": ["region"]}),
        )
        max_results = mloda.run_all(
            [f_max],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        max_table = None
        for table in max_results:
            if isinstance(table, pa.Table) and "value_int__max_grouped" in table.column_names:
                max_table = table
                break
        assert max_table is not None
        max_col = sorted(max_table.column("value_int__max_grouped").to_pylist())
        assert max_col == sorted([20, 60, 40, -10])

        assert min_table.num_rows == max_table.num_rows == 4
