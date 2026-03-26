"""Tests for Pandas frame aggregate implementation."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.pandas_frame_aggregate import (
    PandasFrameAggregate,
)

pytest.importorskip("pandas")


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "region": ["A", "A", "A", "A", "B", "B", "B"],
        "timestamp": [1, 2, 3, 4, 1, 2, 3],
        "value": [10, 20, 30, 40, 100, 200, 300],
    })


class TestPandasRolling:
    def test_rolling_sum_2(self, sample_df: pd.DataFrame) -> None:
        feature = Feature(
            "value__sum_rolling_2",
            options=Options(context={"partition_by": ["region"], "order_by": "timestamp"}),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = PandasFrameAggregate.calculate_feature(sample_df, fs)
        col = result["value__sum_rolling_2"].tolist()

        assert col[0] == 10.0
        assert col[1] == 30.0
        assert col[2] == 50.0
        assert col[3] == 70.0


class TestPandasCumulative:
    def test_cumsum(self, sample_df: pd.DataFrame) -> None:
        feature = Feature(
            "value__cumsum",
            options=Options(context={"partition_by": ["region"], "order_by": "timestamp"}),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = PandasFrameAggregate.calculate_feature(sample_df, fs)
        col = result["value__cumsum"].tolist()

        assert col[0] == 10.0
        assert col[1] == 30.0
        assert col[2] == 60.0
        assert col[3] == 100.0


class TestPandasExpanding:
    def test_expanding_avg(self, sample_df: pd.DataFrame) -> None:
        feature = Feature(
            "value__expanding_avg",
            options=Options(context={"partition_by": ["region"], "order_by": "timestamp"}),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = PandasFrameAggregate.calculate_feature(sample_df, fs)
        col = result["value__expanding_avg"].tolist()

        assert col[0] == 10.0
        assert col[1] == 15.0
        assert col[2] == 20.0
        assert col[3] == 25.0


class TestPandasRowPreserving:
    def test_output_rows_equal_input(self, sample_df: pd.DataFrame) -> None:
        feature = Feature(
            "value__sum_rolling_2",
            options=Options(context={"partition_by": ["region"], "order_by": "timestamp"}),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = PandasFrameAggregate.calculate_feature(sample_df, fs)
        assert len(result) == len(sample_df)
