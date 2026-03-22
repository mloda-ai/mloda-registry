"""Integration tests for window aggregation through mloda's full pipeline.

These tests verify that window aggregation feature groups work end-to-end
through mloda's runtime, including plugin discovery, feature resolution,
and the PluginCollector mechanism.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator import PyArrowDataOpsTestDataCreator
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.pyarrow_window_aggregation import (
    PyArrowWindowAggregation,
)


class TestIntegrationBasic:
    """Test a single window aggregation feature through the full mloda pipeline."""

    def test_sum_groupby_through_pipeline(self) -> None:
        """Run value_int__sum_groupby through run_all and verify the result."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowWindowAggregation}
        )

        feature = Feature(
            "value_int__sum_groupby",
            options=Options(context={"partition_by": ["region"]}),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        # Find the table containing the aggregation result
        result_table = None
        for table in results:
            if isinstance(table, pa.Table) and "value_int__sum_groupby" in table.column_names:
                result_table = table
                break

        assert result_table is not None, "No result table with value_int__sum_groupby found"
        assert result_table.num_rows == 12

        result_col = result_table.column("value_int__sum_groupby").to_pylist()
        expected = [25, 25, 25, 25, 140, 140, 140, 140, 70, 70, 70, -10]
        assert result_col == expected

    def test_avg_groupby_through_pipeline(self) -> None:
        """Run value_int__avg_groupby through run_all and verify the result."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowWindowAggregation}
        )

        feature = Feature(
            "value_int__avg_groupby",
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
            if isinstance(table, pa.Table) and "value_int__avg_groupby" in table.column_names:
                result_table = table
                break

        assert result_table is not None, "No result table with value_int__avg_groupby found"
        assert result_table.num_rows == 12

        result_col = result_table.column("value_int__avg_groupby").to_pylist()
        expected = [6.25, 6.25, 6.25, 6.25, 46.667, 46.667, 46.667, 46.667, 23.333, 23.333, 23.333, -10.0]
        assert result_col == pytest.approx(expected, rel=1e-3)


class TestIntegrationMultipleFeatures:
    """Test multiple window aggregation features in a single run_all call."""

    def test_sum_and_avg_together(self) -> None:
        """Request both sum and avg features in one pipeline run."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowWindowAggregation}
        )

        f_sum = Feature(
            "value_int__sum_groupby",
            options=Options(context={"partition_by": ["region"]}),
        )
        f_avg = Feature(
            "value_int__avg_groupby",
            options=Options(context={"partition_by": ["region"]}),
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
            if "value_int__sum_groupby" in table.column_names:
                sum_col = table.column("value_int__sum_groupby").to_pylist()
                expected_sum = [25, 25, 25, 25, 140, 140, 140, 140, 70, 70, 70, -10]
                assert sum_col == expected_sum
                sum_found = True
            if "value_int__avg_groupby" in table.column_names:
                avg_col = table.column("value_int__avg_groupby").to_pylist()
                expected_avg = [
                    6.25,
                    6.25,
                    6.25,
                    6.25,
                    46.667,
                    46.667,
                    46.667,
                    46.667,
                    23.333,
                    23.333,
                    23.333,
                    -10.0,
                ]
                assert avg_col == pytest.approx(expected_avg, rel=1e-3)
                avg_found = True

        assert sum_found, "sum_groupby result not found in any result table"
        assert avg_found, "avg_groupby result not found in any result table"

    def test_different_aggregation_types(self) -> None:
        """Request min and max features in one pipeline run."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowWindowAggregation}
        )

        f_min = Feature(
            "value_int__min_groupby",
            options=Options(context={"partition_by": ["region"]}),
        )
        f_max = Feature(
            "value_int__max_groupby",
            options=Options(context={"partition_by": ["region"]}),
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
            if "value_int__min_groupby" in table.column_names:
                min_col = table.column("value_int__min_groupby").to_pylist()
                expected_min = [-5, -5, -5, -5, 30, 30, 30, 30, 15, 15, 15, -10]
                assert min_col == expected_min
                min_found = True
            if "value_int__max_groupby" in table.column_names:
                max_col = table.column("value_int__max_groupby").to_pylist()
                expected_max = [20, 20, 20, 20, 60, 60, 60, 60, 40, 40, 40, -10]
                assert max_col == expected_max
                max_found = True

        assert min_found, "min_groupby result not found in any result table"
        assert max_found, "max_groupby result not found in any result table"


class TestIntegrationPluginDiscovery:
    """Test that PluginCollector correctly discovers and filters window aggregation plugins."""

    def test_feature_group_is_discoverable(self) -> None:
        """Verify PyArrowWindowAggregation can be enabled via PluginCollector."""
        plugin_collector = PluginCollector.enabled_feature_groups(
            {PyArrowDataOpsTestDataCreator, PyArrowWindowAggregation}
        )

        assert plugin_collector.applicable_feature_group_class(PyArrowWindowAggregation)
        assert plugin_collector.applicable_feature_group_class(PyArrowDataOpsTestDataCreator)

    def test_disabled_feature_group_blocks_execution(self) -> None:
        """When PyArrowWindowAggregation is not in the enabled set, pipeline should fail."""
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator})

        feature = Feature(
            "value_int__sum_groupby",
            options=Options(context={"partition_by": ["region"]}),
        )

        with pytest.raises(ValueError):
            mloda.run_all(
                [feature],
                compute_frameworks={PyArrowTable},
                plugin_collector=plugin_collector,
            )

    def test_match_feature_group_criteria(self) -> None:
        """Verify that match_feature_group_criteria works for valid window aggregation features."""
        options = Options(context={"partition_by": ["region"]})
        assert PyArrowWindowAggregation.match_feature_group_criteria("value_int__sum_groupby", options)
        assert PyArrowWindowAggregation.match_feature_group_criteria("value_int__avg_groupby", options)
        assert PyArrowWindowAggregation.match_feature_group_criteria("value_int__count_groupby", options)

    def test_match_rejects_invalid_feature_names(self) -> None:
        """Verify that match_feature_group_criteria rejects non-matching feature names."""
        options = Options(context={"partition_by": ["region"]})
        assert not PyArrowWindowAggregation.match_feature_group_criteria("value_int", options)
        assert not PyArrowWindowAggregation.match_feature_group_criteria("value_int__sum", options)

    def test_match_rejects_missing_partition_by(self) -> None:
        """Verify that match_feature_group_criteria rejects when partition_by is missing."""
        options = Options()
        assert not PyArrowWindowAggregation.match_feature_group_criteria("value_int__sum_groupby", options)
