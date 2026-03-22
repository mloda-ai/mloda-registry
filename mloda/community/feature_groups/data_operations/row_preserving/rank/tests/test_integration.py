"""Integration tests for rank through mloda's full pipeline."""

from __future__ import annotations

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.rank.pyarrow_rank import (
    PyArrowRank,
)


class TestIntegrationBasic:
    """Test rank features through the full mloda pipeline."""

    def test_row_number_through_pipeline(self) -> None:
        """Run value_int__row_number_ranked through run_all."""
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, PyArrowRank})

        feature = Feature(
            "value_int__row_number_ranked",
            options=Options(context={"partition_by": ["region"], "order_by": "value_int"}),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        result_table = None
        for table in results:
            if isinstance(table, pa.Table) and "value_int__row_number_ranked" in table.column_names:
                result_table = table
                break

        assert result_table is not None
        assert result_table.num_rows == 12

        result_col = result_table.column("value_int__row_number_ranked").to_pylist()
        expected = [3, 1, 2, 4, 4, 2, 1, 3, 1, 2, 3, 1]
        assert result_col == expected

    def test_dense_rank_through_pipeline(self) -> None:
        """Run value_int__dense_rank_ranked through run_all."""
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, PyArrowRank})

        feature = Feature(
            "value_int__dense_rank_ranked",
            options=Options(context={"partition_by": ["region"], "order_by": "value_int"}),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks={PyArrowTable},
            plugin_collector=plugin_collector,
        )

        result_table = None
        for table in results:
            if isinstance(table, pa.Table) and "value_int__dense_rank_ranked" in table.column_names:
                result_table = table
                break

        assert result_table is not None
        assert result_table.num_rows == 12

        result_col = result_table.column("value_int__dense_rank_ranked").to_pylist()
        expected = [3, 1, 2, 4, 4, 2, 1, 3, 1, 1, 2, 1]
        assert result_col == expected


class TestIntegrationPluginDiscovery:
    """Test plugin discovery for rank feature groups."""

    def test_feature_group_is_discoverable(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, PyArrowRank})
        assert plugin_collector.applicable_feature_group_class(PyArrowRank)

    def test_disabled_feature_group_blocks_execution(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator})

        feature = Feature(
            "value_int__row_number_ranked",
            options=Options(context={"partition_by": ["region"], "order_by": "value_int"}),
        )

        with pytest.raises(ValueError):
            mloda.run_all(
                [feature],
                compute_frameworks={PyArrowTable},
                plugin_collector=plugin_collector,
            )

    def test_match_feature_group_criteria(self) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "value_int"})
        assert PyArrowRank.match_feature_group_criteria("value_int__row_number_ranked", options)
        assert PyArrowRank.match_feature_group_criteria("value_int__rank_ranked", options)

    def test_match_rejects_missing_order_by(self) -> None:
        options = Options(context={"partition_by": ["region"]})
        assert not PyArrowRank.match_feature_group_criteria("value_int__row_number_ranked", options)

    def test_match_rejects_missing_partition_by(self) -> None:
        options = Options(context={"order_by": "value_int"})
        assert not PyArrowRank.match_feature_group_criteria("value_int__row_number_ranked", options)
