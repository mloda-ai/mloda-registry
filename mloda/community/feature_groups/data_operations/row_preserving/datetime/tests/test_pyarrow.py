"""Tests for PyArrowDateTimeExtraction compute implementation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.row_preserving.datetime.pyarrow_datetime import (
    PyArrowDateTimeExtraction,
)


@pytest.fixture
def sample_table() -> pa.Table:
    return pa.table(
        {
            "timestamp": [
                datetime(2024, 1, 15, 10, 30, 45),
                datetime(2024, 6, 22, 14, 0, 0),
                datetime(2024, 12, 25, 0, 0, 0),
                datetime(2024, 3, 9, 8, 15, 30),
                datetime(2024, 7, 13, 18, 45, 59),
            ],
            "value": [1, 2, 3, 4, 5],
        }
    )


def _make_feature_set(feature_name: str) -> FeatureSet:
    feature = Feature(feature_name, options=Options())
    fs = FeatureSet()
    fs.add(feature)
    return fs


class TestYearExtraction:
    def test_year_values(self, sample_table: pa.Table) -> None:
        fs = _make_feature_set("timestamp__year")
        result = PyArrowDateTimeExtraction.calculate_feature(sample_table, fs)

        assert isinstance(result, pa.Table)
        result_col = result.column("timestamp__year").to_pylist()
        assert result_col == [2024, 2024, 2024, 2024, 2024]

    def test_month_values(self, sample_table: pa.Table) -> None:
        fs = _make_feature_set("timestamp__month")
        result = PyArrowDateTimeExtraction.calculate_feature(sample_table, fs)

        result_col = result.column("timestamp__month").to_pylist()
        assert result_col == [1, 6, 12, 3, 7]

    def test_day_values(self, sample_table: pa.Table) -> None:
        fs = _make_feature_set("timestamp__day")
        result = PyArrowDateTimeExtraction.calculate_feature(sample_table, fs)

        result_col = result.column("timestamp__day").to_pylist()
        assert result_col == [15, 22, 25, 9, 13]

    def test_hour_values(self, sample_table: pa.Table) -> None:
        fs = _make_feature_set("timestamp__hour")
        result = PyArrowDateTimeExtraction.calculate_feature(sample_table, fs)

        result_col = result.column("timestamp__hour").to_pylist()
        assert result_col == [10, 14, 0, 8, 18]

    def test_minute_values(self, sample_table: pa.Table) -> None:
        fs = _make_feature_set("timestamp__minute")
        result = PyArrowDateTimeExtraction.calculate_feature(sample_table, fs)

        result_col = result.column("timestamp__minute").to_pylist()
        assert result_col == [30, 0, 0, 15, 45]

    def test_second_values(self, sample_table: pa.Table) -> None:
        fs = _make_feature_set("timestamp__second")
        result = PyArrowDateTimeExtraction.calculate_feature(sample_table, fs)

        result_col = result.column("timestamp__second").to_pylist()
        assert result_col == [45, 0, 0, 30, 59]


class TestIsWeekend:
    def test_is_weekend_values(self, sample_table: pa.Table) -> None:
        """2024-01-15=Mon, 2024-06-22=Sat, 2024-12-25=Wed, 2024-03-09=Sat, 2024-07-13=Sat."""
        fs = _make_feature_set("timestamp__is_weekend")
        result = PyArrowDateTimeExtraction.calculate_feature(sample_table, fs)

        result_col = result.column("timestamp__is_weekend").to_pylist()
        assert result_col == [0, 1, 0, 1, 1]

    def test_dayofweek_values(self, sample_table: pa.Table) -> None:
        """2024-01-15=Mon(0), 2024-06-22=Sat(5), 2024-12-25=Wed(2), 2024-03-09=Sat(5), 2024-07-13=Sat(5)."""
        fs = _make_feature_set("timestamp__dayofweek")
        result = PyArrowDateTimeExtraction.calculate_feature(sample_table, fs)

        result_col = result.column("timestamp__dayofweek").to_pylist()
        assert result_col == [0, 5, 2, 5, 5]


class TestQuarter:
    def test_quarter_values(self, sample_table: pa.Table) -> None:
        """Jan=Q1, Jun=Q2, Dec=Q4, Mar=Q1, Jul=Q3."""
        fs = _make_feature_set("timestamp__quarter")
        result = PyArrowDateTimeExtraction.calculate_feature(sample_table, fs)

        result_col = result.column("timestamp__quarter").to_pylist()
        assert result_col == [1, 2, 4, 1, 3]


class TestRowPreserving:
    def test_output_rows_equal_input_rows(self, sample_table: pa.Table) -> None:
        fs = _make_feature_set("timestamp__year")
        result = PyArrowDateTimeExtraction.calculate_feature(sample_table, fs)

        assert result.num_rows == sample_table.num_rows

    def test_result_type(self, sample_table: pa.Table) -> None:
        fs = _make_feature_set("timestamp__year")
        result = PyArrowDateTimeExtraction.calculate_feature(sample_table, fs)

        assert isinstance(result, pa.Table)

    def test_new_column_added(self, sample_table: pa.Table) -> None:
        fs = _make_feature_set("timestamp__year")
        result = PyArrowDateTimeExtraction.calculate_feature(sample_table, fs)

        assert "timestamp__year" in result.column_names
