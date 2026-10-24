"""Integration tests for offset through mloda's full pipeline.

PyArrow has no native window functions, so offset runs on pandas here: the group is served by
PandasOffset. The tests verify that offset operations work end-to-end through mloda's
runtime, including plugin discovery, feature resolution, and PluginCollector.
"""

from __future__ import annotations

from typing import Any

import pytest
from mloda.user import Feature, FeatureResolutionError, Options, PluginCollector, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.offset.base import (
    OffsetFeatureGroup,
)
from mloda.testing.data_creator.base import column_selective
from mloda.testing.data_creator.pandas import PandasDataOpsTestDataCreator
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.helpers import (
    assert_values_with_nulls,
    extract_column,
    result_column_names,
)


@pytest.fixture
def pd() -> Any:
    """pandas, or a skip of the requesting test when it is not installed."""
    return pytest.importorskip("pandas")


needs_pandas = pytest.mark.usefixtures("pd")


def _offset_column(name: str, creator: type = PandasDataOpsTestDataCreator, order_by: str = "value_int") -> list[Any]:
    """Run one offset feature on pandas through ``run_all`` and return its column."""
    pytest.importorskip("pandas")
    plugin_collector = PluginCollector.enabled_feature_groups({creator, OffsetFeatureGroup})
    feature = Feature(name, options=Options(context={"partition_by": ["region"], "order_by": order_by}))

    results = mloda.run_all([feature], compute_frameworks=[PandasDataFrame], plugin_collector=plugin_collector)

    result_frame = next(f for f in results if name in result_column_names(f))
    column = extract_column(result_frame, name)
    assert len(column) == 12
    return column


class TestIntegrationBasic:
    def test_lag_through_pipeline(self) -> None:
        assert len(_offset_column("value_int__lag_1_offset")) == 12

    def test_first_value_through_pipeline(self) -> None:
        assert len(_offset_column("value_int__first_value_offset")) == 12

    def test_lag_with_column_selective_source(self) -> None:
        creator = column_selective(PandasDataOpsTestDataCreator)
        result = _offset_column("value_int__lag_1_offset", creator=creator, order_by="timestamp")
        expected = [None, 10, -5, 0, None, None, 50, 30, None, 15, 15, None]
        assert_values_with_nulls(result, expected, nan_is_null=True)


class TestIntegrationPluginDiscovery:
    def test_feature_group_is_discoverable(self) -> None:
        plugin_collector = PluginCollector.enabled_feature_groups({PandasDataOpsTestDataCreator, OffsetFeatureGroup})
        assert plugin_collector.applicable_feature_group_class(OffsetFeatureGroup)

    def test_match_feature_group_criteria(self) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "value_int"})
        assert OffsetFeatureGroup.match_feature_group_criteria("value_int__lag_1_offset", options)
        assert OffsetFeatureGroup.match_feature_group_criteria("value_int__first_value_offset", options)

    def test_match_rejects_missing_order_by(self) -> None:
        options = Options(context={"partition_by": ["region"]})
        assert not OffsetFeatureGroup.match_feature_group_criteria("value_int__lag_1_offset", options)

    def test_match_rejects_empty_partition_by(self) -> None:
        options = Options(context={"partition_by": [], "order_by": "value_int"})
        assert not OffsetFeatureGroup.match_feature_group_criteria("value_int__lag_1_offset", options)


class TestIntegrationSeveralFrameworks:
    """One OffsetFeatureGroup is served by one Implementation per framework, chosen per feature."""

    @staticmethod
    def _feature(name: str) -> Feature:
        return Feature(name, options=Options(context={"partition_by": ["region"], "order_by": "value_int"}))

    def test_several_frameworks_enabled_runs_lag(self, pd: Any) -> None:
        pytest.importorskip("polars")
        from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

        name = "value_int__lag_1_offset"
        plugin_collector = PluginCollector.enabled_feature_groups({PandasDataOpsTestDataCreator, OffsetFeatureGroup})

        results = mloda.run_all(
            [self._feature(name)],
            compute_frameworks=[PandasDataFrame, PolarsLazyDataFrame],
            plugin_collector=plugin_collector,
        )

        frames = [f for f in results if isinstance(f, pd.DataFrame) and name in f.columns]
        assert len(frames) == 1
        assert len(frames[0]) == 12

    def test_pyarrow_alone_is_a_resolution_error(self) -> None:
        """No offset Implementation serves PyArrow, so the group is unresolvable there."""
        plugin_collector = PluginCollector.enabled_feature_groups({PyArrowDataOpsTestDataCreator, OffsetFeatureGroup})

        with pytest.raises(FeatureResolutionError):
            mloda.run_all(
                [self._feature("value_int__lag_1_offset")],
                compute_frameworks=[PyArrowTable],
                plugin_collector=plugin_collector,
            )
