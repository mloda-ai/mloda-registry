"""Tests for PyArrowWindowAggregation compute implementation."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.pyarrow_window_aggregation import (
    PyArrowWindowAggregation,
)
from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation import (
    WindowAggregationTestBase,
    make_feature_set,
)


class TestPyArrowWindowAggregation(WindowAggregationTestBase):
    """Standard tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowWindowAggregation

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        return arrow_table

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        return list(result.column(column_name).to_pylist())

    def get_row_count(self, result: Any) -> int:
        return int(result.num_rows)

    def get_expected_type(self) -> Any:
        return pa.Table


class TestPyArrowStatisticalAggregations:
    """Test std, var, median aggregations."""

    def test_std_groupby_region(self, sample_data: pa.Table) -> None:
        """Standard deviation of value_int partitioned by region."""
        feature_set = make_feature_set("value_int__std_groupby", ["region"])
        result = PyArrowWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, pa.Table)
        assert result.num_rows == 12
        assert "value_int__std_groupby" in result.column_names

        result_col = result.column("value_int__std_groupby").to_pylist()
        # Group A values: [10, -5, 0, 20], group B non-null: [50, 30, 60]
        a_vals = [10, -5, 0, 20]
        a_std = (sum((x - 6.25) ** 2 for x in a_vals) / (len(a_vals) - 1)) ** 0.5
        assert result_col[0] == pytest.approx(a_std, rel=1e-6)
        assert result_col[1] == pytest.approx(a_std, rel=1e-6)
        assert result_col[2] == pytest.approx(a_std, rel=1e-6)
        assert result_col[3] == pytest.approx(a_std, rel=1e-6)

    def test_var_groupby_region(self, sample_data: pa.Table) -> None:
        """Variance of value_int partitioned by region."""
        feature_set = make_feature_set("value_int__var_groupby", ["region"])
        result = PyArrowWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, pa.Table)
        assert result.num_rows == 12
        assert "value_int__var_groupby" in result.column_names

        result_col = result.column("value_int__var_groupby").to_pylist()
        # Group A: var([10, -5, 0, 20]) sample variance
        a_vals = [10, -5, 0, 20]
        a_var = sum((x - 6.25) ** 2 for x in a_vals) / (len(a_vals) - 1)
        assert result_col[0] == pytest.approx(a_var, rel=1e-6)
        assert result_col[1] == pytest.approx(a_var, rel=1e-6)

    def test_median_groupby_region(self, sample_data: pa.Table) -> None:
        """Median of value_int partitioned by region."""
        feature_set = make_feature_set("value_int__median_groupby", ["region"])
        result = PyArrowWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, pa.Table)
        assert result.num_rows == 12
        assert "value_int__median_groupby" in result.column_names

        result_col = result.column("value_int__median_groupby").to_pylist()
        # Group A sorted: [-5, 0, 10, 20] -> median = (0+10)/2 = 5.0
        assert result_col[0] == pytest.approx(5.0, rel=1e-6)
        assert result_col[1] == pytest.approx(5.0, rel=1e-6)
        # Group B sorted (non-null): [30, 50, 60] -> median = 50
        assert result_col[4] == pytest.approx(50.0, rel=1e-6)
        # Group C sorted: [15, 15, 40] -> median = 15
        assert result_col[8] == pytest.approx(15.0, rel=1e-6)
        # None group: [-10] -> median = -10
        assert result_col[11] == pytest.approx(-10.0, rel=1e-6)


class TestPyArrowAdvancedAggregations:
    """Test mode, nunique, first, last aggregations."""

    def test_mode_groupby_region(self, sample_data: pa.Table) -> None:
        """Mode of value_int partitioned by region."""
        feature_set = make_feature_set("value_int__mode_groupby", ["region"])
        result = PyArrowWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, pa.Table)
        assert result.num_rows == 12
        assert "value_int__mode_groupby" in result.column_names

        result_col = result.column("value_int__mode_groupby").to_pylist()
        # Group C has 15 appearing twice, so mode = 15
        assert result_col[8] == 15
        assert result_col[9] == 15
        assert result_col[10] == 15
        # None group: single value -10
        assert result_col[11] == -10

    def test_nunique_groupby_region(self, sample_data: pa.Table) -> None:
        """Count of unique non-null value_int values partitioned by region."""
        feature_set = make_feature_set("value_int__nunique_groupby", ["region"])
        result = PyArrowWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, pa.Table)
        assert result.num_rows == 12
        assert "value_int__nunique_groupby" in result.column_names

        result_col = result.column("value_int__nunique_groupby").to_pylist()
        # Group A: {10, -5, 0, 20} -> 4 unique
        assert result_col[0] == 4
        # Group B: {50, 30, 60} -> 3 unique (null skipped)
        assert result_col[4] == 3
        # Group C: {15, 15, 40} -> 2 unique
        assert result_col[8] == 2
        # None group: {-10} -> 1 unique
        assert result_col[11] == 1

    def test_first_groupby_region(self, sample_data: pa.Table) -> None:
        """First value of value_int partitioned by region (first non-null or first row)."""
        feature_set = make_feature_set("value_int__first_groupby", ["region"])
        result = PyArrowWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, pa.Table)
        assert result.num_rows == 12
        assert "value_int__first_groupby" in result.column_names

        result_col = result.column("value_int__first_groupby").to_pylist()
        # Group A first value: 10
        assert result_col[0] == 10
        assert result_col[1] == 10
        assert result_col[2] == 10
        assert result_col[3] == 10
        # None group first value: -10
        assert result_col[11] == -10

    def test_last_groupby_region(self, sample_data: pa.Table) -> None:
        """Last value of value_int partitioned by region."""
        feature_set = make_feature_set("value_int__last_groupby", ["region"])
        result = PyArrowWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, pa.Table)
        assert result.num_rows == 12
        assert "value_int__last_groupby" in result.column_names

        result_col = result.column("value_int__last_groupby").to_pylist()
        # Group A last value: 20
        assert result_col[0] == 20
        assert result_col[1] == 20
        assert result_col[2] == 20
        assert result_col[3] == 20
        # Group B last value: 60
        assert result_col[4] == 60
        # Group C last value: 40
        assert result_col[8] == 40
        # None group last value: -10
        assert result_col[11] == -10


class TestPyArrowNullHandling:
    """Additional null handling edge cases specific to PyArrow."""

    def test_ec021_all_null_column_aggregation(self, sample_data: pa.Table) -> None:
        """EC-021: score column is all null. Aggregation should produce all nulls."""
        feature_set = make_feature_set("score__sum_groupby", ["region"])
        result = PyArrowWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, pa.Table)
        assert result.num_rows == 12

        result_col = result.column("score__sum_groupby").to_pylist()
        # All values should be None since the source column is entirely null
        assert all(v is None for v in result_col)


class TestPyArrowMultiKeyPartition:
    """Test partitioning by multiple keys (e.g. region + category)."""

    def test_multi_key_partition_sum(self, sample_data: pa.Table) -> None:
        """Sum of value_int partitioned by [region, category]."""
        feature_set = make_feature_set("value_int__sum_groupby", ["region", "category"])
        result = PyArrowWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, pa.Table)
        assert result.num_rows == 12

        result_col = result.column("value_int__sum_groupby").to_pylist()
        # region="A", category="X": rows 0, 2 -> value_int [10, 0] -> sum = 10
        assert result_col[0] == 10
        assert result_col[2] == 10
        # region="A", category="Y": rows 1, 3 -> value_int [-5, 20] -> sum = 15
        assert result_col[1] == 15
        assert result_col[3] == 15
        # region="B", category="X": rows 4, 7 -> value_int [None, 60] -> sum = 60
        assert result_col[4] == 60
        assert result_col[7] == 60
        # region="B", category="Y": row 5 -> value_int [50] -> sum = 50
        assert result_col[5] == 50
        # region="B", category=None: row 6 -> value_int [30] -> sum = 30
        assert result_col[6] == 30

    def test_multi_key_partition_count(self, sample_data: pa.Table) -> None:
        """Count of non-null value_int partitioned by [region, category]."""
        feature_set = make_feature_set("value_int__count_groupby", ["region", "category"])
        result = PyArrowWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, pa.Table)
        assert result.num_rows == 12

        result_col = result.column("value_int__count_groupby").to_pylist()
        # region="A", category="X": rows 0, 2 -> 2 non-null
        assert result_col[0] == 2
        assert result_col[2] == 2
        # region="A", category="Y": rows 1, 3 -> 2 non-null
        assert result_col[1] == 2
        assert result_col[3] == 2
        # region="B", category="X": rows 4, 7 -> 1 non-null (row 4 is null)
        assert result_col[4] == 1
        assert result_col[7] == 1

    def test_multi_key_float_column(self, sample_data: pa.Table) -> None:
        """Avg of value_float partitioned by [region, category] on float data."""
        feature_set = make_feature_set("value_float__avg_groupby", ["region", "category"])
        result = PyArrowWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, pa.Table)
        assert result.num_rows == 12

        result_col = result.column("value_float__avg_groupby").to_pylist()
        # region="A", category="X": rows 0, 2 -> value_float [1.5, None] -> avg = 1.5
        assert result_col[0] == pytest.approx(1.5, rel=1e-6)
        assert result_col[2] == pytest.approx(1.5, rel=1e-6)
        # region="A", category="Y": rows 1, 3 -> value_float [2.5, 0.0] -> avg = 1.25
        assert result_col[1] == pytest.approx(1.25, rel=1e-6)
        assert result_col[3] == pytest.approx(1.25, rel=1e-6)
