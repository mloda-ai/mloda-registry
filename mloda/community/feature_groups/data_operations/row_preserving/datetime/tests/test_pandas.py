"""Tests for PandasDateTimeExtraction compute implementation."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("pandas")

import pandas as pd

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.row_preserving.datetime.pandas_datetime import (
    PandasDateTimeExtraction,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    datetime(2024, 1, 15, 10, 30, 45),
                    datetime(2024, 6, 22, 14, 0, 0),
                    datetime(2024, 12, 25, 0, 0, 0),
                    datetime(2024, 3, 9, 8, 15, 30),
                    datetime(2024, 7, 13, 18, 45, 59),
                ]
            ),
            "value": [1, 2, 3, 4, 5],
        }
    )


def _make_feature_set(feature_name: str) -> FeatureSet:
    feature = Feature(feature_name, options=Options())
    fs = FeatureSet()
    fs.add(feature)
    return fs


class TestYearExtraction:
    def test_year_values(self, sample_df: pd.DataFrame) -> None:
        fs = _make_feature_set("timestamp__year")
        result = PandasDateTimeExtraction.calculate_feature(sample_df, fs)

        assert isinstance(result, pd.DataFrame)
        result_col = result["timestamp__year"].tolist()
        assert result_col == [2024, 2024, 2024, 2024, 2024]

    def test_month_values(self, sample_df: pd.DataFrame) -> None:
        fs = _make_feature_set("timestamp__month")
        result = PandasDateTimeExtraction.calculate_feature(sample_df, fs)

        result_col = result["timestamp__month"].tolist()
        assert result_col == [1, 6, 12, 3, 7]

    def test_day_values(self, sample_df: pd.DataFrame) -> None:
        fs = _make_feature_set("timestamp__day")
        result = PandasDateTimeExtraction.calculate_feature(sample_df, fs)

        result_col = result["timestamp__day"].tolist()
        assert result_col == [15, 22, 25, 9, 13]


class TestIsWeekend:
    def test_is_weekend_values(self, sample_df: pd.DataFrame) -> None:
        fs = _make_feature_set("timestamp__is_weekend")
        result = PandasDateTimeExtraction.calculate_feature(sample_df, fs)

        result_col = result["timestamp__is_weekend"].tolist()
        assert result_col == [0, 1, 0, 1, 1]

    def test_dayofweek_values(self, sample_df: pd.DataFrame) -> None:
        fs = _make_feature_set("timestamp__dayofweek")
        result = PandasDateTimeExtraction.calculate_feature(sample_df, fs)

        result_col = result["timestamp__dayofweek"].tolist()
        assert result_col == [0, 5, 2, 5, 5]


class TestRowPreserving:
    def test_output_rows_equal_input_rows(self, sample_df: pd.DataFrame) -> None:
        fs = _make_feature_set("timestamp__year")
        result = PandasDateTimeExtraction.calculate_feature(sample_df, fs)

        assert len(result) == 5

    def test_result_type(self, sample_df: pd.DataFrame) -> None:
        fs = _make_feature_set("timestamp__year")
        result = PandasDateTimeExtraction.calculate_feature(sample_df, fs)

        assert isinstance(result, pd.DataFrame)
