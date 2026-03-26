"""Tests for PyArrowStringOps compute implementation."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.string.pyarrow_string import (
    PyArrowStringOps,
)


@pytest.fixture
def sample_table() -> pa.Table:
    return pa.table(
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
    def test_upper_values(self, sample_table: pa.Table) -> None:
        fs = _make_feature_set("name__upper")
        result = PyArrowStringOps.calculate_feature(sample_table, fs)

        assert isinstance(result, pa.Table)
        result_col = result.column("name__upper").to_pylist()
        assert result_col == ["ALICE", "BOB", "CHARLIE", None, "EVE"]


class TestLower:
    def test_lower_values(self, sample_table: pa.Table) -> None:
        fs = _make_feature_set("name__lower")
        result = PyArrowStringOps.calculate_feature(sample_table, fs)

        result_col = result.column("name__lower").to_pylist()
        assert result_col == ["alice", "bob", "charlie", None, "eve"]


class TestLength:
    def test_length_values(self, sample_table: pa.Table) -> None:
        fs = _make_feature_set("name__length")
        result = PyArrowStringOps.calculate_feature(sample_table, fs)

        result_col = result.column("name__length").to_pylist()
        assert result_col == [5, 3, 7, None, 3]


class TestNullPropagation:
    def test_null_produces_null_for_upper(self, sample_table: pa.Table) -> None:
        fs = _make_feature_set("name__upper")
        result = PyArrowStringOps.calculate_feature(sample_table, fs)

        result_col = result.column("name__upper").to_pylist()
        assert result_col[3] is None

    def test_null_produces_null_for_length(self, sample_table: pa.Table) -> None:
        fs = _make_feature_set("name__length")
        result = PyArrowStringOps.calculate_feature(sample_table, fs)

        result_col = result.column("name__length").to_pylist()
        assert result_col[3] is None


class TestRowPreserving:
    def test_output_rows_equal_input_rows(self, sample_table: pa.Table) -> None:
        fs = _make_feature_set("name__upper")
        result = PyArrowStringOps.calculate_feature(sample_table, fs)

        assert result.num_rows == sample_table.num_rows

    def test_result_type(self, sample_table: pa.Table) -> None:
        fs = _make_feature_set("name__upper")
        result = PyArrowStringOps.calculate_feature(sample_table, fs)

        assert isinstance(result, pa.Table)

    def test_new_column_added(self, sample_table: pa.Table) -> None:
        fs = _make_feature_set("name__upper")
        result = PyArrowStringOps.calculate_feature(sample_table, fs)

        assert "name__upper" in result.column_names
