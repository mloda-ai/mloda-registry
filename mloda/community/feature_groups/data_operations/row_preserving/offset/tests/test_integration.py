"""Integration tests for offset through mloda's full pipeline."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator.pandas import PandasDataOpsTestDataCreator
from mloda.user import Feature, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.offset.pandas_offset import PandasOffset


class TestIntegrationBasic:
    def test_lag_through_pipeline(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({PandasDataOpsTestDataCreator, PandasOffset})

        feature = Feature(
            "value_int__lag_1_offset",
            options=Options(context={"partition_by": ["region"], "order_by": "value_int"}),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        result_df = None
        for table in results:
            if isinstance(table, pd.DataFrame) and "value_int__lag_1_offset" in table.columns:
                result_df = table
                break

        assert result_df is not None
        assert len(result_df) == 12

    def test_first_value_through_pipeline(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({PandasDataOpsTestDataCreator, PandasOffset})

        feature = Feature(
            "value_int__first_value_offset",
            options=Options(context={"partition_by": ["region"], "order_by": "value_int"}),
        )

        results = mloda.run_all(
            [feature],
            compute_frameworks={PandasDataFrame},
            plugin_collector=plugin_collector,
        )

        result_df = None
        for table in results:
            if isinstance(table, pd.DataFrame) and "value_int__first_value_offset" in table.columns:
                result_df = table
                break

        assert result_df is not None
        assert len(result_df) == 12


class TestIntegrationPluginDiscovery:
    def test_feature_group_is_discoverable(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({PandasDataOpsTestDataCreator, PandasOffset})
        assert plugin_collector.applicable_feature_group_class(PandasOffset)

    def test_match_feature_group_criteria(self) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "value_int"})
        assert PandasOffset.match_feature_group_criteria("value_int__lag_1_offset", options)
        assert PandasOffset.match_feature_group_criteria("value_int__first_value_offset", options)

    def test_match_rejects_missing_order_by(self) -> None:
        options = Options(context={"partition_by": ["region"]})
        assert not PandasOffset.match_feature_group_criteria("value_int__lag_1_offset", options)
