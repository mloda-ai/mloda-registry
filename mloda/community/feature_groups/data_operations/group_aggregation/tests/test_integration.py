"""Integration tests for group aggregation through mloda's full pipeline.

These tests verify that group aggregation feature groups work end-to-end
through mloda's runtime, including plugin discovery, feature resolution,
and the PluginCollector mechanism.

Note: mloda's pipeline only returns requested feature columns, so
integration tests verify values as sets (not keyed by partition columns).
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.group_aggregation.pyarrow_group_aggregation import (
    PyArrowGroupAggregation,
)


class TestIntegrationBasic:
    """Test a single group aggregation feature through the full mloda pipeline."""

    def test_sum_grouped_through_pipeline(self) -> None:
        """Run value_int__sum_grouped through run_all and verify the result."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowGroupAggregation}
        )

        feature = Feature(
            "value_int__sum_grouped",
            options=Options(context={"partition_by": ["region"]}),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        result_table = None
        for table in results:
            if isinstance(table, pa.Table) and "value_int__sum_grouped" in table.column_names:
                result_table = table
                break

        assert result_table is not None, "No result table with value_int__sum_grouped found"
        assert result_table.num_rows == 4

        result_col = result_table.column("value_int__sum_grouped").to_pylist()
        assert sorted(result_col) == sorted([25, 140, 70, -10])

    def test_avg_grouped_through_pipeline(self) -> None:
        """Run value_int__avg_grouped through run_all and verify the result."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowGroupAggregation}
        )

        feature = Feature(
            "value_int__avg_grouped",
            options=Options(context={"partition_by": ["region"]}),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        result_table = None
        for table in results:
            if isinstance(table, pa.Table) and "value_int__avg_grouped" in table.column_names:
                result_table = table
                break

        assert result_table is not None, "No result table with value_int__avg_grouped found"
        assert result_table.num_rows == 4

        result_col = sorted(result_table.column("value_int__avg_grouped").to_pylist())
        expected = sorted([6.25, 46.667, 23.333, -10.0])
        assert result_col == pytest.approx(expected, rel=1e-3)


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


class TestIntegrationPluginDiscovery:
    """Test that PluginCollector correctly discovers and filters group aggregation plugins."""

    def test_feature_group_is_discoverable(self) -> None:
        """Verify PyArrowGroupAggregation can be enabled via PluginCollector."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowGroupAggregation}
        )

        assert plugin_collector.applicable_feature_group_class(PyArrowGroupAggregation)
        assert plugin_collector.applicable_feature_group_class(PyArrowDataOpsTestDataCreator)

    def test_disabled_feature_group_blocks_execution(self) -> None:
        """When PyArrowGroupAggregation is not in the enabled set, pipeline should fail."""
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator})

        feature = Feature(
            "value_int__sum_grouped",
            options=Options(context={"partition_by": ["region"]}),
        )

        with pytest.raises(ValueError):
            mloda.run_all(
                [feature],
                compute_frameworks={PyArrowTable},
                plugin_collector=plugin_collector,
            )

    def test_match_feature_group_criteria(self) -> None:
        """Verify that match_feature_group_criteria works for valid group aggregation features."""
        options = Options(context={"partition_by": ["region"]})
        assert PyArrowGroupAggregation.match_feature_group_criteria("value_int__sum_grouped", options)
        assert PyArrowGroupAggregation.match_feature_group_criteria("value_int__avg_grouped", options)
        assert PyArrowGroupAggregation.match_feature_group_criteria("value_int__count_grouped", options)

    def test_match_rejects_invalid_feature_names(self) -> None:
        """Verify that match_feature_group_criteria rejects non-matching feature names."""
        options = Options(context={"partition_by": ["region"]})
        assert not PyArrowGroupAggregation.match_feature_group_criteria("value_int", options)
        assert not PyArrowGroupAggregation.match_feature_group_criteria("value_int__sum", options)

    def test_match_rejects_missing_partition_by(self) -> None:
        """Verify that match_feature_group_criteria rejects when partition_by is missing."""
        options = Options()
        assert not PyArrowGroupAggregation.match_feature_group_criteria("value_int__sum_grouped", options)
