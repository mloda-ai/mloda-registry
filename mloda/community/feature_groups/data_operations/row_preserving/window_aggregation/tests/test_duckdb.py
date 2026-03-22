"""Tests for DuckdbWindowAggregation compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

duckdb = pytest.importorskip("duckdb")

from mloda.testing.data_creator import PyArrowDataOpsTestDataCreator
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.duckdb_window_aggregation import (
    DuckdbWindowAggregation,
)
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.tests.conftest import (
    make_feature_set,
)


@pytest.fixture
def conn() -> Any:
    connection = duckdb.connect()
    yield connection
    connection.close()


@pytest.fixture
def sample_data(conn: Any) -> DuckdbRelation:
    arrow_table = PyArrowDataOpsTestDataCreator.create()
    return DuckdbRelation.from_arrow(conn, arrow_table)


def _extract_column(result: DuckdbRelation, column_name: str) -> list[Any]:
    """Extract a column from a DuckdbRelation as a Python list."""
    return list(result.to_arrow_table().column(column_name).to_pylist())


class TestDuckdbBasicAggregations:
    """Test sum, avg, count, min, max with pre-computed expected values."""

    def test_sum_groupby_region(self, sample_data: DuckdbRelation) -> None:
        """Sum of value_int partitioned by region, broadcast back to every row."""
        feature_set = make_feature_set("value_int__sum_groupby", ["region"])
        result = DuckdbWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, DuckdbRelation)

        result_col = _extract_column(result, "value_int__sum_groupby")
        expected = [25, 25, 25, 25, 140, 140, 140, 140, 70, 70, 70, -10]
        assert result_col == expected

    def test_avg_groupby_region(self, sample_data: DuckdbRelation) -> None:
        """Average of value_int partitioned by region, broadcast back to every row."""
        feature_set = make_feature_set("value_int__avg_groupby", ["region"])
        result = DuckdbWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, DuckdbRelation)

        result_col = _extract_column(result, "value_int__avg_groupby")
        expected = [6.25, 6.25, 6.25, 6.25, 46.667, 46.667, 46.667, 46.667, 23.333, 23.333, 23.333, -10.0]
        assert result_col == pytest.approx(expected, rel=1e-3)

    def test_count_groupby_region(self, sample_data: DuckdbRelation) -> None:
        """Count of non-null value_int partitioned by region."""
        feature_set = make_feature_set("value_int__count_groupby", ["region"])
        result = DuckdbWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, DuckdbRelation)

        result_col = _extract_column(result, "value_int__count_groupby")
        expected = [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 1]
        assert result_col == expected

    def test_min_groupby_region(self, sample_data: DuckdbRelation) -> None:
        """Minimum of value_int partitioned by region."""
        feature_set = make_feature_set("value_int__min_groupby", ["region"])
        result = DuckdbWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, DuckdbRelation)

        result_col = _extract_column(result, "value_int__min_groupby")
        expected = [-5, -5, -5, -5, 30, 30, 30, 30, 15, 15, 15, -10]
        assert result_col == expected

    def test_max_groupby_region(self, sample_data: DuckdbRelation) -> None:
        """Maximum of value_int partitioned by region."""
        feature_set = make_feature_set("value_int__max_groupby", ["region"])
        result = DuckdbWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, DuckdbRelation)

        result_col = _extract_column(result, "value_int__max_groupby")
        expected = [20, 20, 20, 20, 60, 60, 60, 60, 40, 40, 40, -10]
        assert result_col == expected


class TestDuckdbNullHandling:
    """Test null handling edge cases."""

    def test_ec016_avg_with_null_values_in_group(self, sample_data: DuckdbRelation) -> None:
        """EC-016: Group B has a null value_int at row 4. Avg should skip it."""
        feature_set = make_feature_set("value_int__avg_groupby", ["region"])
        result = DuckdbWindowAggregation.calculate_feature(sample_data, feature_set)

        result_col = _extract_column(result, "value_int__avg_groupby")
        # Group B: (50+30+60)/3 = 46.667 (null at row 4 skipped)
        b_expected = 140.0 / 3.0
        assert result_col[4] == pytest.approx(b_expected, rel=1e-6)
        assert result_col[5] == pytest.approx(b_expected, rel=1e-6)
        assert result_col[6] == pytest.approx(b_expected, rel=1e-6)
        assert result_col[7] == pytest.approx(b_expected, rel=1e-6)

    def test_ec019_null_group_key_forms_own_group(self, sample_data: DuckdbRelation) -> None:
        """EC-019: Row 11 has region=None. It should form its own group."""
        feature_set = make_feature_set("value_int__sum_groupby", ["region"])
        result = DuckdbWindowAggregation.calculate_feature(sample_data, feature_set)

        result_col = _extract_column(result, "value_int__sum_groupby")
        # Row 11 (region=None, value_int=-10) is its own group
        assert result_col[11] == -10


class TestDuckdbRowPreservation:
    """Test that output row count always equals input row count."""

    def test_output_rows_equal_input_rows(self, sample_data: DuckdbRelation) -> None:
        """Output must have exactly 12 rows, same as input."""
        feature_set = make_feature_set("value_int__sum_groupby", ["region"])
        result = DuckdbWindowAggregation.calculate_feature(sample_data, feature_set)

        arrow_result = result.to_arrow_table()
        assert arrow_result.num_rows == 12

    def test_result_is_duckdb_relation(self, sample_data: DuckdbRelation) -> None:
        """The result of calculate_feature must be a DuckdbRelation."""
        feature_set = make_feature_set("value_int__min_groupby", ["region"])
        result = DuckdbWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, DuckdbRelation)
