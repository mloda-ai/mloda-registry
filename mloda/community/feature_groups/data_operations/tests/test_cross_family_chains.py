"""Chains that span data-operation families resolve and compute end to end through ``mloda.run_all``."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest
from mloda.user import Feature, Options, mloda
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.aggregation.base import AggregationFeatureGroup
from mloda.community.feature_groups.data_operations.row_changing.resample.base import ResampleFeatureGroup
from mloda.community.feature_groups.data_operations.row_preserving.offset.base import OffsetFeatureGroup
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
    WindowAggregationFeatureGroup,
)
from mloda.testing.data_creator.pandas import PandasDataOpsTestDataCreator
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.integration import enabled_with_implementations


def _sort_none_last(values: list[Any]) -> list[Any]:
    return sorted(values, key=lambda x: (x is None, x))


def _column(results: list[Any], name: str) -> list[Any]:
    """Values of ``name`` from the single result frame holding it, None for null or NaN."""
    found = []
    for t in results:
        if isinstance(t, pa.Table) and name in t.column_names:
            found.append(t.column(name).to_pylist())
        elif hasattr(t, "columns") and name in t.columns:
            found.append([None if v != v else v for v in t[name].tolist()])
    assert len(found) == 1, f"expected one frame holding {name}, found {len(found)}"
    return list(found[0])


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
        assert _sort_none_last(_column(results, name)) == pytest.approx([4.0, 16.36, 100.0, None])

    def test_window_aggregation_over_aggregation(self) -> None:
        name = "value_int__sum_agg__sum_window"
        options = Options(context=self._CONTEXT, cascade_context_keys=frozenset(self._CONTEXT))
        plugin_collector = enabled_with_implementations(
            {PyArrowDataOpsTestDataCreator, AggregationFeatureGroup, WindowAggregationFeatureGroup}
        )
        results = mloda.run_all(
            [Feature(name, options=options)], compute_frameworks=[PyArrowTable], plugin_collector=plugin_collector
        )
        assert _sort_none_last(_column(results, name)) == [-10, 25, 70, 140]

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
        assert _sort_none_last(_column(results, name)) == pytest.approx(expected)
