"""Tests for PolarsLazyStringOps compute implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("polars")

import polars as pl

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.string.polars_lazy_string import (
    PolarsLazyStringOps,
)


@pytest.fixture
def sample_lf() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", None, "Eve"],
            "value": [1, 2, 3, 4, 5],
        }
    ).lazy()


def _make_feature_set(feature_name: str) -> FeatureSet:
    feature = Feature(feature_name, options=Options())
    fs = FeatureSet()
    fs.add(feature)
    return fs


class TestUpper:
    def test_upper_values(self, sample_lf: pl.LazyFrame) -> None:
        fs = _make_feature_set("name__upper")
        result = PolarsLazyStringOps.calculate_feature(sample_lf, fs)

        assert isinstance(result, pl.LazyFrame)
        result_col = result.collect()["name__upper"].to_list()
        assert result_col == ["ALICE", "BOB", "CHARLIE", None, "EVE"]


class TestLength:
    def test_length_values(self, sample_lf: pl.LazyFrame) -> None:
        fs = _make_feature_set("name__length")
        result = PolarsLazyStringOps.calculate_feature(sample_lf, fs)

        result_col = result.collect()["name__length"].to_list()
        assert result_col == [5, 3, 7, None, 3]


class TestNullPropagation:
    def test_null_produces_null_for_upper(self, sample_lf: pl.LazyFrame) -> None:
        fs = _make_feature_set("name__upper")
        result = PolarsLazyStringOps.calculate_feature(sample_lf, fs)

        result_col = result.collect()["name__upper"].to_list()
        assert result_col[3] is None

    def test_null_produces_null_for_length(self, sample_lf: pl.LazyFrame) -> None:
        fs = _make_feature_set("name__length")
        result = PolarsLazyStringOps.calculate_feature(sample_lf, fs)

        result_col = result.collect()["name__length"].to_list()
        assert result_col[3] is None


class TestRowPreserving:
    def test_output_rows_equal_input_rows(self, sample_lf: pl.LazyFrame) -> None:
        fs = _make_feature_set("name__upper")
        result = PolarsLazyStringOps.calculate_feature(sample_lf, fs)

        assert result.collect().height == 5

    def test_result_type(self, sample_lf: pl.LazyFrame) -> None:
        fs = _make_feature_set("name__upper")
        result = PolarsLazyStringOps.calculate_feature(sample_lf, fs)

        assert isinstance(result, pl.LazyFrame)
