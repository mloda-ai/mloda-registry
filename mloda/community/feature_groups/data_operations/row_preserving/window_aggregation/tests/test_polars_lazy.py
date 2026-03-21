"""Tests for PolarsLazyWindowAggregation compute implementation."""

from __future__ import annotations

import pyarrow as pa
import pytest

pytest.importorskip("polars")

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator import PyArrowDataOpsTestDataCreator
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.polars_lazy_window_aggregation import (
    PolarsLazyWindowAggregation,
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


class TestPolarsLazyBasicAggregations:
    """Test sum, avg, count, min, max with pre-computed expected values."""

    def test_sum_groupby_region(self, sample_data: pa.Table) -> None:
        """Sum of value_int partitioned by region, broadcast back to every row."""
        feature_set = _make_feature_set("value_int__sum_groupby", ["region"])
        result = PolarsLazyWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, pa.Table)
        assert result.num_rows == 12

        result_col = result.column("value_int__sum_groupby").to_pylist()
        expected = [25, 25, 25, 25, 140, 140, 140, 140, 70, 70, 70, -10]
        assert result_col == expected

    def test_avg_groupby_region(self, sample_data: pa.Table) -> None:
        """Average of value_int partitioned by region, broadcast back to every row."""
        feature_set = _make_feature_set("value_int__avg_groupby", ["region"])
        result = PolarsLazyWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, pa.Table)
        assert result.num_rows == 12

        result_col = result.column("value_int__avg_groupby").to_pylist()
        expected = [6.25, 6.25, 6.25, 6.25, 46.667, 46.667, 46.667, 46.667, 23.333, 23.333, 23.333, -10.0]
        assert result_col == pytest.approx(expected, rel=1e-3)

    def test_count_groupby_region(self, sample_data: pa.Table) -> None:
        """Count of non-null value_int partitioned by region."""
        feature_set = _make_feature_set("value_int__count_groupby", ["region"])
        result = PolarsLazyWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, pa.Table)
        assert result.num_rows == 12

        result_col = result.column("value_int__count_groupby").to_pylist()
        expected = [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 1]
        assert result_col == expected

    def test_min_groupby_region(self, sample_data: pa.Table) -> None:
        """Minimum of value_int partitioned by region."""
        feature_set = _make_feature_set("value_int__min_groupby", ["region"])
        result = PolarsLazyWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, pa.Table)
        assert result.num_rows == 12

        result_col = result.column("value_int__min_groupby").to_pylist()
        expected = [-5, -5, -5, -5, 30, 30, 30, 30, 15, 15, 15, -10]
        assert result_col == expected

    def test_max_groupby_region(self, sample_data: pa.Table) -> None:
        """Maximum of value_int partitioned by region."""
        feature_set = _make_feature_set("value_int__max_groupby", ["region"])
        result = PolarsLazyWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, pa.Table)
        assert result.num_rows == 12

        result_col = result.column("value_int__max_groupby").to_pylist()
        expected = [20, 20, 20, 20, 60, 60, 60, 60, 40, 40, 40, -10]
        assert result_col == expected


class TestPolarsLazyNullHandling:
    """Test null handling edge cases."""

    def test_ec016_avg_with_null_values_in_group(self, sample_data: pa.Table) -> None:
        """EC-016: Group B has a null value_int at row 4. Avg should skip it."""
        feature_set = _make_feature_set("value_int__avg_groupby", ["region"])
        result = PolarsLazyWindowAggregation.calculate_feature(sample_data, feature_set)

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
        result = PolarsLazyWindowAggregation.calculate_feature(sample_data, feature_set)

        result_col = result.column("value_int__sum_groupby").to_pylist()
        # Row 11 (region=None, value_int=-10) is its own group
        assert result_col[11] == -10


class TestPolarsLazyRowPreservation:
    """Test that output row count always equals input row count."""

    def test_output_rows_equal_input_rows(self, sample_data: pa.Table) -> None:
        """Output must have exactly 12 rows, same as input."""
        feature_set = _make_feature_set("value_int__sum_groupby", ["region"])
        result = PolarsLazyWindowAggregation.calculate_feature(sample_data, feature_set)

        assert result.num_rows == sample_data.num_rows
        assert result.num_rows == 12

    def test_original_columns_preserved(self, sample_data: pa.Table) -> None:
        """All original columns should still be present in the result."""
        feature_set = _make_feature_set("value_int__sum_groupby", ["region"])
        result = PolarsLazyWindowAggregation.calculate_feature(sample_data, feature_set)

        for col_name in sample_data.column_names:
            assert col_name in result.column_names, f"Original column '{col_name}' missing from result"

    def test_result_is_pyarrow_table(self, sample_data: pa.Table) -> None:
        """The result of calculate_feature must be a PyArrow Table."""
        feature_set = _make_feature_set("value_int__min_groupby", ["region"])
        result = PolarsLazyWindowAggregation.calculate_feature(sample_data, feature_set)

        assert isinstance(result, pa.Table)
