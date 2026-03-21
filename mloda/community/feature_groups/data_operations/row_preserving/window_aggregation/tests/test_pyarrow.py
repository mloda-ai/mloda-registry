"""Tests for PyArrowWindowAggregation compute implementation."""

from __future__ import annotations

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator import PyArrowDataOpsTestDataCreator
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.pyarrow_window_aggregation import (
    PyArrowWindowAggregation,
)


@pytest.fixture
def sample_data() -> pa.Table:
    """Return the shared 12-row test dataset as a PyArrow Table."""
    return PyArrowDataOpsTestDataCreator.create()


def _make_feature_set(feature_name: str, partition_by: list[str]) -> FeatureSet:
    """Helper to build a FeatureSet with partition_by options."""
    feature = Feature(
        feature_name,
        options=Options(context={"partition_by": partition_by}),
    )
    fs = FeatureSet()
    fs.add(feature)
    return fs


class TestPyArrowBasicAggregations:
    """Test sum, avg, count, min, max with pre-computed expected values."""

    def test_sum_groupby_region(self, sample_data: pa.Table) -> None:
        """Sum of value_int partitioned by region, broadcast back to every row."""
        feature_set = _make_feature_set("value_int__sum_groupby", ["region"])
        result = PyArrowWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, pa.Table)
        assert result.num_rows == 12

        result_col = result.column("value_int__sum_groupby").to_pylist()
        expected = [25, 25, 25, 25, 140, 140, 140, 140, 70, 70, 70, -10]
        assert result_col == expected

    def test_avg_groupby_region(self, sample_data: pa.Table) -> None:
        """Average of value_int partitioned by region, broadcast back to every row."""
        feature_set = _make_feature_set("value_int__avg_groupby", ["region"])
        result = PyArrowWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, pa.Table)
        assert result.num_rows == 12

        result_col = result.column("value_int__avg_groupby").to_pylist()
        expected = [6.25, 6.25, 6.25, 6.25, 46.667, 46.667, 46.667, 46.667, 23.333, 23.333, 23.333, -10.0]
        assert result_col == pytest.approx(expected, rel=1e-3)

    def test_count_groupby_region(self, sample_data: pa.Table) -> None:
        """Count of non-null value_int partitioned by region."""
        feature_set = _make_feature_set("value_int__count_groupby", ["region"])
        result = PyArrowWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, pa.Table)
        assert result.num_rows == 12

        result_col = result.column("value_int__count_groupby").to_pylist()
        expected = [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 1]
        assert result_col == expected

    def test_min_groupby_region(self, sample_data: pa.Table) -> None:
        """Minimum of value_int partitioned by region."""
        feature_set = _make_feature_set("value_int__min_groupby", ["region"])
        result = PyArrowWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, pa.Table)
        assert result.num_rows == 12

        result_col = result.column("value_int__min_groupby").to_pylist()
        expected = [-5, -5, -5, -5, 30, 30, 30, 30, 15, 15, 15, -10]
        assert result_col == expected

    def test_max_groupby_region(self, sample_data: pa.Table) -> None:
        """Maximum of value_int partitioned by region."""
        feature_set = _make_feature_set("value_int__max_groupby", ["region"])
        result = PyArrowWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, pa.Table)
        assert result.num_rows == 12

        result_col = result.column("value_int__max_groupby").to_pylist()
        expected = [20, 20, 20, 20, 60, 60, 60, 60, 40, 40, 40, -10]
        assert result_col == expected


class TestPyArrowStatisticalAggregations:
    """Test std, var, median aggregations."""

    def test_std_groupby_region(self, sample_data: pa.Table) -> None:
        """Standard deviation of value_int partitioned by region."""
        feature_set = _make_feature_set("value_int__std_groupby", ["region"])
        result = PyArrowWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, pa.Table)
        assert result.num_rows == 12
        assert "value_int__std_groupby" in result.column_names

        result_col = result.column("value_int__std_groupby").to_pylist()
        # All values in each group should be identical (broadcast)
        # Group A values: [10, -5, 0, 20], group B non-null: [50, 30, 60]
        a_vals = [10, -5, 0, 20]
        a_std = (sum((x - 6.25) ** 2 for x in a_vals) / (len(a_vals) - 1)) ** 0.5
        assert result_col[0] == pytest.approx(a_std, rel=1e-6)
        assert result_col[1] == pytest.approx(a_std, rel=1e-6)
        assert result_col[2] == pytest.approx(a_std, rel=1e-6)
        assert result_col[3] == pytest.approx(a_std, rel=1e-6)

    def test_var_groupby_region(self, sample_data: pa.Table) -> None:
        """Variance of value_int partitioned by region."""
        feature_set = _make_feature_set("value_int__var_groupby", ["region"])
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
        feature_set = _make_feature_set("value_int__median_groupby", ["region"])
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
        feature_set = _make_feature_set("value_int__mode_groupby", ["region"])
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
        feature_set = _make_feature_set("value_int__nunique_groupby", ["region"])
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
        feature_set = _make_feature_set("value_int__first_groupby", ["region"])
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
        feature_set = _make_feature_set("value_int__last_groupby", ["region"])
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
    """Test null handling edge cases."""

    def test_ec016_avg_with_null_values_in_group(self, sample_data: pa.Table) -> None:
        """EC-016: Group B has a null value_int at row 4. Avg should skip it."""
        feature_set = _make_feature_set("value_int__avg_groupby", ["region"])
        result = PyArrowWindowAggregation.calculate_feature(sample_data, feature_set)

        result_col = result.column("value_int__avg_groupby").to_pylist()
        # Group B: (50+30+60)/3 = 46.667 (null at row 4 skipped)
        b_expected = 140.0 / 3.0
        assert result_col[4] == pytest.approx(b_expected, rel=1e-6)
        assert result_col[5] == pytest.approx(b_expected, rel=1e-6)
        assert result_col[6] == pytest.approx(b_expected, rel=1e-6)
        assert result_col[7] == pytest.approx(b_expected, rel=1e-6)

    def test_ec019_null_group_key_forms_own_group(self, sample_data: pa.Table) -> None:
        """EC-019: Row 11 has region=None. It should form its own group."""
        feature_set = _make_feature_set("value_int__sum_groupby", ["region"])
        result = PyArrowWindowAggregation.calculate_feature(sample_data, feature_set)

        result_col = result.column("value_int__sum_groupby").to_pylist()
        # Row 11 (region=None, value_int=-10) is its own group
        assert result_col[11] == -10

    def test_ec021_all_null_column_aggregation(self, sample_data: pa.Table) -> None:
        """EC-021: score column is all null. Aggregation should produce all nulls."""
        feature_set = _make_feature_set("score__sum_groupby", ["region"])
        result = PyArrowWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, pa.Table)
        assert result.num_rows == 12

        result_col = result.column("score__sum_groupby").to_pylist()
        # All values should be None since the source column is entirely null
        assert all(v is None for v in result_col)


class TestPyArrowRowPreservation:
    """Test that output row count always equals input row count."""

    def test_output_rows_equal_input_rows(self, sample_data: pa.Table) -> None:
        """Output must have exactly 12 rows, same as input."""
        feature_set = _make_feature_set("value_int__sum_groupby", ["region"])
        result = PyArrowWindowAggregation.calculate_feature(sample_data, feature_set)

        assert result.num_rows == sample_data.num_rows
        assert result.num_rows == 12

    def test_original_columns_preserved(self, sample_data: pa.Table) -> None:
        """All original columns should still be present in the result."""
        feature_set = _make_feature_set("value_int__sum_groupby", ["region"])
        result = PyArrowWindowAggregation.calculate_feature(sample_data, feature_set)

        for col_name in sample_data.column_names:
            assert col_name in result.column_names, f"Original column '{col_name}' missing from result"

    def test_new_column_added(self, sample_data: pa.Table) -> None:
        """The aggregation result column should be added to the table."""
        feature_set = _make_feature_set("value_int__max_groupby", ["region"])
        result = PyArrowWindowAggregation.calculate_feature(sample_data, feature_set)

        assert "value_int__max_groupby" in result.column_names

    def test_result_is_pyarrow_table(self, sample_data: pa.Table) -> None:
        """The result of calculate_feature must be a PyArrow Table."""
        feature_set = _make_feature_set("value_int__min_groupby", ["region"])
        result = PyArrowWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, pa.Table)


class TestPyArrowMultiKeyPartition:
    """Test partitioning by multiple keys (e.g. region + category)."""

    def test_multi_key_partition_sum(self, sample_data: pa.Table) -> None:
        """Sum of value_int partitioned by [region, category]."""
        feature_set = _make_feature_set("value_int__sum_groupby", ["region", "category"])
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
        feature_set = _make_feature_set("value_int__count_groupby", ["region", "category"])
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
        feature_set = _make_feature_set("value_float__avg_groupby", ["region", "category"])
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
