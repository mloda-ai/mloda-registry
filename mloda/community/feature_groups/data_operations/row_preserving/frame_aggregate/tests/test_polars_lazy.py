"""Tests for Polars lazy frame aggregate implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

pl = pytest.importorskip("polars")

if TYPE_CHECKING:
    import polars

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.polars_lazy_frame_aggregate import (
    PolarsLazyFrameAggregate,
)


@pytest.fixture()
def polars_data() -> polars.LazyFrame:
    return pl.DataFrame(
        {
            "region": ["A", "A", "A", "A", "B", "B", "B"],
            "timestamp": [1, 2, 3, 4, 1, 2, 3],
            "value": [10, 20, 30, 40, 100, 200, 300],
        }
    ).lazy()


class TestPolarsRolling:
    def test_rolling_sum_2(self, polars_data: polars.LazyFrame) -> None:
        feature = Feature(
            "value__sum_rolling_2",
            options=Options(context={"partition_by": ["region"], "order_by": "timestamp"}),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = PolarsLazyFrameAggregate.calculate_feature(polars_data, fs)
        df = result.collect()
        col = df["value__sum_rolling_2"].to_list()

        assert col[0] == 10
        assert col[1] == 30
        assert col[2] == 50
        assert col[3] == 70


class TestPolarsCumulative:
    def test_cumsum(self, polars_data: polars.LazyFrame) -> None:
        feature = Feature(
            "value__cumsum",
            options=Options(context={"partition_by": ["region"], "order_by": "timestamp"}),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = PolarsLazyFrameAggregate.calculate_feature(polars_data, fs)
        df = result.collect()
        col = df["value__cumsum"].to_list()

        assert col[0] == 10
        assert col[1] == 30
        assert col[2] == 60
        assert col[3] == 100
