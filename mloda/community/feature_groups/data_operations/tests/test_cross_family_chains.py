"""Chains that span data-operation families resolve and compute end to end through ``mloda.run_all``."""

from __future__ import annotations

from typing import Any

import pytest
from mloda.provider import FeatureGroup
from mloda.user import Feature, Options, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.aggregation.base import AggregationFeatureGroup
from mloda.community.feature_groups.data_operations.row_changing.resample.base import ResampleFeatureGroup
from mloda.community.feature_groups.data_operations.row_preserving.offset.base import OffsetFeatureGroup
from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.base import (
    ScalarAggregateFeatureGroup,
)
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
    WindowAggregationFeatureGroup,
)
from mloda.testing.data_creator.pandas import PandasDataOpsTestDataCreator
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.helpers import result_column_values, sort_none_last
from mloda.testing.feature_groups.data_operations.integration import enabled_with_implementations


def _column(results: list[Any], name: str) -> list[Any]:
    """Values of ``name`` from the single result frame holding it, None for null or NaN."""
    found = result_column_values(results, name)
    assert len(found) == 1, f"expected one frame holding {name}, found {len(found)}"
    return found[0]


class TestCrossFamilyChains:
    """Chains across groups that compute today; each is one reducing or row-preserving step per name segment."""

    _CONTEXT = {"partition_by": ["region"]}

    def test_aggregation_over_resample(self) -> None:
        name = "value_float__resample_1_day_sum__sum_agg"
        context = {"time_column": "timestamp", "partition_by": ["region"]}
        options = Options(context=context, cascade_context_keys=frozenset(context))
        plugin_collector = enabled_with_implementations(
            {PyArrowDataOpsTestDataCreator, ResampleFeatureGroup, AggregationFeatureGroup}
        )
        results = mloda.run_all(
            [Feature(name, options=options)], compute_frameworks=[PyArrowTable], plugin_collector=plugin_collector
        )
        assert sort_none_last(_column(results, name)) == pytest.approx([4.0, 16.36, 100.0, None])

    def test_window_aggregation_over_aggregation(self) -> None:
        name = "value_int__sum_agg__sum_window"
        options = Options(context=self._CONTEXT, cascade_context_keys=frozenset(self._CONTEXT))
        plugin_collector = enabled_with_implementations(
            {PyArrowDataOpsTestDataCreator, AggregationFeatureGroup, WindowAggregationFeatureGroup}
        )
        results = mloda.run_all(
            [Feature(name, options=options)], compute_frameworks=[PyArrowTable], plugin_collector=plugin_collector
        )
        assert sort_none_last(_column(results, name)) == [-10, 25, 70, 140]

    def test_window_aggregation_over_window_aggregation_over_aggregation(self) -> None:
        name = "value_int__sum_agg__sum_window__max_window"
        options = Options(context=self._CONTEXT, cascade_context_keys=frozenset(self._CONTEXT))
        plugin_collector = enabled_with_implementations(
            {PyArrowDataOpsTestDataCreator, AggregationFeatureGroup, WindowAggregationFeatureGroup}
        )
        results = mloda.run_all(
            [Feature(name, options=options)], compute_frameworks=[PyArrowTable], plugin_collector=plugin_collector
        )
        assert sort_none_last(_column(results, name)) == [-10, 25, 70, 140]

    def test_offset_over_resample(self) -> None:
        name = "value_float__resample_1_day_sum__lag_1_offset"
        context = {"time_column": "timestamp", "partition_by": ["region"], "order_by": "timestamp"}
        options = Options(context=context, cascade_context_keys=frozenset(context))
        plugin_collector = enabled_with_implementations(
            {PandasDataOpsTestDataCreator, ResampleFeatureGroup, OffsetFeatureGroup}
        )
        results = mloda.run_all(
            [Feature(name, options=options)], compute_frameworks=[PandasDataFrame], plugin_collector=plugin_collector
        )
        expected = [1e-15, 1.5, 2.36, 2.5, 6.5, 100.0, None, None, None, None, None]
        assert sort_none_last(_column(results, name)) == pytest.approx(expected)

    def test_offset_over_offset_over_resample(self) -> None:
        name = "value_float__resample_1_day_sum__lag_1_offset__lag_1_offset"
        context = {"time_column": "timestamp", "partition_by": ["region"], "order_by": "timestamp"}
        options = Options(context=context, cascade_context_keys=frozenset(context))
        plugin_collector = enabled_with_implementations(
            {PandasDataOpsTestDataCreator, ResampleFeatureGroup, OffsetFeatureGroup}
        )
        results = mloda.run_all(
            [Feature(name, options=options)], compute_frameworks=[PandasDataFrame], plugin_collector=plugin_collector
        )
        expected = [1e-15, 1.5, 2.36, 2.5, None, None, None, None, None, None, None]
        assert sort_none_last(_column(results, name)) == pytest.approx(expected)


class TestReducingNextToRowPreserving:
    """A reducing feature and a row-preserving feature from another family in one request both compute."""

    def test_aggregation_next_to_window_aggregation(self) -> None:
        options = Options(context={"partition_by": ["region"]})
        features = [Feature("value_int__sum_agg", options=options), Feature("value_int__sum_window", options=options)]
        plugin_collector = enabled_with_implementations(
            {PandasDataOpsTestDataCreator, AggregationFeatureGroup, WindowAggregationFeatureGroup}
        )
        results = mloda.run_all(features, compute_frameworks=[PandasDataFrame], plugin_collector=plugin_collector)

        assert sort_none_last(_column(results, "value_int__sum_agg")) == [-10, 25, 70, 140]
        window = _column(results, "value_int__sum_window")
        assert len(window) == 12
        assert set(window) == {-10, 25, 70, 140}


@pytest.mark.parametrize(
    "group, expected",
    [
        (AggregationFeatureGroup, True),
        (ResampleFeatureGroup, True),
        (WindowAggregationFeatureGroup, False),
        (ScalarAggregateFeatureGroup, False),
    ],
)
def test_replaces_rows_only_for_reducing_groups(group: type[FeatureGroup], expected: bool) -> None:
    assert group.replaces_rows() is expected
