"""Tests for PandasStringOps compute implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("pandas")

import pandas as pd

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.string.pandas_string import (
    PandasStringOps,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", None, "Eve"],
            "value": [1, 2, 3, 4, 5],
        }
    )


def _make_feature_set(feature_name: str) -> FeatureSet:
    feature = Feature(feature_name, options=Options())
    fs = FeatureSet()
    fs.add(feature)
    return fs


class TestUpper:
    def test_upper_values(self, sample_df: pd.DataFrame) -> None:
        fs = _make_feature_set("name__upper")
        result = PandasStringOps.calculate_feature(sample_df, fs)

        assert isinstance(result, pd.DataFrame)
        result_col = result["name__upper"].tolist()
        assert result_col[:3] == ["ALICE", "BOB", "CHARLIE"]
        assert pd.isna(result_col[3])
        assert result_col[4] == "EVE"


class TestLength:
    def test_length_values(self, sample_df: pd.DataFrame) -> None:
        fs = _make_feature_set("name__length")
        result = PandasStringOps.calculate_feature(sample_df, fs)

        result_col = result["name__length"].tolist()
        assert result_col[0] == 5
        assert result_col[1] == 3
        assert result_col[2] == 7
        assert pd.isna(result_col[3])
        assert result_col[4] == 3


class TestNullPropagation:
    def test_null_produces_null_for_upper(self, sample_df: pd.DataFrame) -> None:
        fs = _make_feature_set("name__upper")
        result = PandasStringOps.calculate_feature(sample_df, fs)

        assert pd.isna(result["name__upper"].iloc[3])

    def test_null_produces_null_for_length(self, sample_df: pd.DataFrame) -> None:
        fs = _make_feature_set("name__length")
        result = PandasStringOps.calculate_feature(sample_df, fs)

        assert pd.isna(result["name__length"].iloc[3])


class TestRowPreserving:
    def test_output_rows_equal_input_rows(self, sample_df: pd.DataFrame) -> None:
        fs = _make_feature_set("name__upper")
        result = PandasStringOps.calculate_feature(sample_df, fs)

        assert len(result) == 5

    def test_result_type(self, sample_df: pd.DataFrame) -> None:
        fs = _make_feature_set("name__upper")
        result = PandasStringOps.calculate_feature(sample_df, fs)

        assert isinstance(result, pd.DataFrame)
