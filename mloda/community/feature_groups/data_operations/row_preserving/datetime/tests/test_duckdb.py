"""Tests for DuckdbDateTimeExtraction compute implementation."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pytest

duckdb = pytest.importorskip("duckdb")

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.user import Feature
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

from mloda.community.feature_groups.data_operations.row_preserving.datetime.duckdb_datetime import (
    DuckdbDateTimeExtraction,
)


@pytest.fixture
def sample_relation() -> Any:
    conn = duckdb.connect()
    arrow_table = pa.table(
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
    relation = DuckdbRelation.from_arrow(conn, arrow_table)
    yield relation
    conn.close()


def _make_feature_set(feature_name: str) -> FeatureSet:
    feature = Feature(feature_name, options=Options())
    fs = FeatureSet()
    fs.add(feature)
    return fs


class TestYearExtraction:
    def test_year_values(self, sample_relation: Any) -> None:
        fs = _make_feature_set("timestamp__year")
        result = DuckdbDateTimeExtraction.calculate_feature(sample_relation, fs)

        result_col = result.to_arrow_table().column("timestamp__year").to_pylist()
        assert result_col == [2024, 2024, 2024, 2024, 2024]

    def test_month_values(self, sample_relation: Any) -> None:
        fs = _make_feature_set("timestamp__month")
        result = DuckdbDateTimeExtraction.calculate_feature(sample_relation, fs)

        result_col = result.to_arrow_table().column("timestamp__month").to_pylist()
        assert result_col == [1, 6, 12, 3, 7]

    def test_day_values(self, sample_relation: Any) -> None:
        fs = _make_feature_set("timestamp__day")
        result = DuckdbDateTimeExtraction.calculate_feature(sample_relation, fs)

        result_col = result.to_arrow_table().column("timestamp__day").to_pylist()
        assert result_col == [15, 22, 25, 9, 13]


class TestIsWeekend:
    def test_is_weekend_values(self, sample_relation: Any) -> None:
        fs = _make_feature_set("timestamp__is_weekend")
        result = DuckdbDateTimeExtraction.calculate_feature(sample_relation, fs)

        result_col = result.to_arrow_table().column("timestamp__is_weekend").to_pylist()
        assert result_col == [0, 1, 0, 1, 1]

    def test_dayofweek_values(self, sample_relation: Any) -> None:
        fs = _make_feature_set("timestamp__dayofweek")
        result = DuckdbDateTimeExtraction.calculate_feature(sample_relation, fs)

        result_col = result.to_arrow_table().column("timestamp__dayofweek").to_pylist()
        assert result_col == [0, 5, 2, 5, 5]


class TestRowPreserving:
    def test_output_rows_equal_input_rows(self, sample_relation: Any) -> None:
        fs = _make_feature_set("timestamp__year")
        result = DuckdbDateTimeExtraction.calculate_feature(sample_relation, fs)

        assert result.to_arrow_table().num_rows == 5

    def test_result_type(self, sample_relation: Any) -> None:
        fs = _make_feature_set("timestamp__year")
        result = DuckdbDateTimeExtraction.calculate_feature(sample_relation, fs)

        assert isinstance(result, DuckdbRelation)
