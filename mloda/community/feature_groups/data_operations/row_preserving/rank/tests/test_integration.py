"""Integration tests for rank through mloda's full pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

pd = pytest.importorskip("pandas")

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator.pandas import PandasDataOpsTestDataCreator
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.rank.pandas_rank import (
    PandasRank,
)

if TYPE_CHECKING:
    import pandas


def _extract_column(df: pandas.DataFrame, col: str) -> list[Any]:
    """Extract a column as a Python list with None for NaN."""
    return [None if pd.isna(v) else v for v in df[col].tolist()]


class TestIntegrationBasic:
    """Test rank features through the full mloda pipeline."""

    def test_row_number_through_pipeline(self) -> None:
        """Run value_int__row_number_ranked through run_all."""
        plugin_collector = PluginCollector.enabled_feature_groups({PandasDataOpsTestDataCreator, PandasRank})

        feature = Feature(
            "value_int__row_number_ranked",
            options=Options(context={"partition_by": ["region"], "order_by": "value_int"}),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        assert len(results) >= 1

        result_df = None
        for table in results:
            if isinstance(table, pd.DataFrame) and "value_int__row_number_ranked" in table.columns:
                result_df = table
                break

        assert result_df is not None
        assert len(result_df) == 12

        result_col = _extract_column(result_df, "value_int__row_number_ranked")
        expected = [3, 1, 2, 4, 4, 2, 1, 3, 1, 2, 3, 1]
        assert result_col == expected

    def test_dense_rank_through_pipeline(self) -> None:
        """Run value_int__dense_rank_ranked through run_all."""
        plugin_collector = PluginCollector.enabled_feature_groups({PandasDataOpsTestDataCreator, PandasRank})

        feature = Feature(
            "value_int__dense_rank_ranked",
            options=Options(context={"partition_by": ["region"], "order_by": "value_int"}),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        result_df = None
        for table in results:
            if isinstance(table, pd.DataFrame) and "value_int__dense_rank_ranked" in table.columns:
                result_df = table
                break

        assert result_df is not None
        assert len(result_df) == 12

        result_col = _extract_column(result_df, "value_int__dense_rank_ranked")
        expected = [3, 1, 2, 4, 4, 2, 1, 3, 1, 1, 2, 1]
        assert result_col == expected

    def test_top_n_through_pipeline(self) -> None:
        """Run value_int__top_3_ranked through run_all."""
        plugin_collector = PluginCollector.enabled_feature_groups({PandasDataOpsTestDataCreator, PandasRank})

        feature = Feature(
            "value_int__top_3_ranked",
            options=Options(context={"partition_by": ["region"], "order_by": "value_int"}),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        result_df = None
        for table in results:
            if isinstance(table, pd.DataFrame) and "value_int__top_3_ranked" in table.columns:
                result_df = table
                break

        assert result_df is not None
        assert len(result_df) == 12

        result_col = _extract_column(result_df, "value_int__top_3_ranked")
        expected = [True, False, True, True, False, True, True, True, True, True, True, True]
        assert result_col == expected


class TestIntegrationPluginDiscovery:
    """Test plugin discovery for rank feature groups."""

    def test_feature_group_is_discoverable(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({PandasDataOpsTestDataCreator, PandasRank})
        assert plugin_collector.applicable_feature_group_class(PandasRank)

    def test_disabled_feature_group_blocks_execution(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({PandasDataOpsTestDataCreator})

        feature = Feature(
            "value_int__row_number_ranked",
            options=Options(context={"partition_by": ["region"], "order_by": "value_int"}),
        )

        with pytest.raises(ValueError):
            mloda.run_all(
                [feature],
                compute_frameworks={PandasDataFrame},
                plugin_collector=plugin_collector,
            )

    def test_match_feature_group_criteria(self) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "value_int"})
        assert PandasRank.match_feature_group_criteria("value_int__row_number_ranked", options)
        assert PandasRank.match_feature_group_criteria("value_int__rank_ranked", options)

    def test_match_rejects_missing_order_by(self) -> None:
        options = Options(context={"partition_by": ["region"]})
        assert not PandasRank.match_feature_group_criteria("value_int__row_number_ranked", options)

    def test_match_rejects_missing_partition_by(self) -> None:
        options = Options(context={"order_by": "value_int"})
        assert not PandasRank.match_feature_group_criteria("value_int__row_number_ranked", options)
