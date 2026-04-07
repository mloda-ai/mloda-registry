"""Shared test base class and expected values for single-column aggregate broadcast tests.

Each test verifies that a scalar aggregate (sum, min, max, etc.) is computed
over a single source column and the result is broadcast to every row.

Expected values are computed from the canonical 12-row dataset in
DataOperationsTestDataCreator, using the 'value_int' column.

value_int = [10, -5, 0, 20, None, 50, 30, 60, 15, 15, 40, -10]
Non-null values: [10, -5, 0, 20, 50, 30, 60, 15, 15, 40, -10] (11 values)
"""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda.user import Feature


# ---------------------------------------------------------------------------
# Expected values (module-level constants)
# ---------------------------------------------------------------------------

# value_int non-null: [10, -5, 0, 20, 50, 30, 60, 15, 15, 40, -10]
EXPECTED_SUM: int = 225
EXPECTED_MIN: int = -10
EXPECTED_MAX: int = 60
EXPECTED_COUNT: int = 11
EXPECTED_AVG: float = 225.0 / 11.0  # ~20.4545...

# Population std/var (ddof=0, matching PyArrow)
_non_null = [10, -5, 0, 20, 50, 30, 60, 15, 15, 40, -10]
_mean = sum(_non_null) / len(_non_null)
EXPECTED_VAR: float = sum((x - _mean) ** 2 for x in _non_null) / len(_non_null)
EXPECTED_STD: float = EXPECTED_VAR**0.5

# Sample std/var (ddof=1)
EXPECTED_VAR_SAMP: float = sum((x - _mean) ** 2 for x in _non_null) / (len(_non_null) - 1)
EXPECTED_STD_SAMP: float = EXPECTED_VAR_SAMP**0.5

# Median of sorted non-null: [-10, -5, 0, 10, 15, 15, 20, 30, 40, 50, 60] -> 15.0
EXPECTED_MEDIAN: float = 15.0


# ---------------------------------------------------------------------------
# Reusable test base class
# ---------------------------------------------------------------------------


class ScalarAggregateTestBase(DataOpsTestBase):
    """Abstract base class for scalar aggregate framework tests."""

    ALL_AGG_TYPES = {
        "sum",
        "min",
        "max",
        "avg",
        "mean",
        "count",
        "std",
        "var",
        "std_pop",
        "std_samp",
        "var_pop",
        "var_samp",
        "median",
    }

    @classmethod
    def supported_agg_types(cls) -> set[str]:
        """Override to restrict supported types for a framework."""
        return cls.ALL_AGG_TYPES

    @classmethod
    def reference_implementation_class(cls) -> Any:
        from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.pyarrow_scalar_aggregate import (
            PyArrowScalarAggregate,
        )

        return PyArrowScalarAggregate

    # -- Concrete tests ------------------------------------------------------

    def test_sum_scalar(self) -> None:
        """Sum of value_int broadcast to all rows."""
        fs = make_feature_set("value_int__sum_scalar")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__sum_scalar")
        assert all(v == EXPECTED_SUM for v in result_col)

    def test_min_scalar(self) -> None:
        fs = make_feature_set("value_int__min_scalar")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__min_scalar")
        assert all(v == EXPECTED_MIN for v in result_col)

    def test_max_scalar(self) -> None:
        fs = make_feature_set("value_int__max_scalar")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__max_scalar")
        assert all(v == EXPECTED_MAX for v in result_col)

    def test_avg_scalar(self) -> None:
        fs = make_feature_set("value_int__avg_scalar")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__avg_scalar")
        assert all(v == pytest.approx(EXPECTED_AVG, rel=1e-6) for v in result_col)

    def test_count_scalar(self) -> None:
        fs = make_feature_set("value_int__count_scalar")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__count_scalar")
        assert all(v == EXPECTED_COUNT for v in result_col)

    def test_std_scalar(self) -> None:
        self._skip_if_unsupported("std")
        fs = make_feature_set("value_int__std_scalar")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__std_scalar")
        assert all(v == pytest.approx(EXPECTED_STD, rel=1e-4) for v in result_col)

    def test_var_scalar(self) -> None:
        self._skip_if_unsupported("var")
        fs = make_feature_set("value_int__var_scalar")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__var_scalar")
        assert all(v == pytest.approx(EXPECTED_VAR, rel=1e-4) for v in result_col)

    def test_std_pop_scalar(self) -> None:
        """Explicit population standard deviation (ddof=0), identical to std."""
        self._skip_if_unsupported("std_pop")
        fs = make_feature_set("value_int__std_pop_scalar")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__std_pop_scalar")
        assert all(v == pytest.approx(EXPECTED_STD, rel=1e-4) for v in result_col)

    def test_std_samp_scalar(self) -> None:
        """Sample standard deviation (ddof=1)."""
        self._skip_if_unsupported("std_samp")
        fs = make_feature_set("value_int__std_samp_scalar")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__std_samp_scalar")
        assert all(v == pytest.approx(EXPECTED_STD_SAMP, rel=1e-4) for v in result_col)

    def test_var_pop_scalar(self) -> None:
        """Explicit population variance (ddof=0), identical to var."""
        self._skip_if_unsupported("var_pop")
        fs = make_feature_set("value_int__var_pop_scalar")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__var_pop_scalar")
        assert all(v == pytest.approx(EXPECTED_VAR, rel=1e-4) for v in result_col)

    def test_var_samp_scalar(self) -> None:
        """Sample variance (ddof=1)."""
        self._skip_if_unsupported("var_samp")
        fs = make_feature_set("value_int__var_samp_scalar")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__var_samp_scalar")
        assert all(v == pytest.approx(EXPECTED_VAR_SAMP, rel=1e-4) for v in result_col)

    def test_median_scalar(self) -> None:
        self._skip_if_unsupported("median")
        fs = make_feature_set("value_int__median_scalar")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__median_scalar")
        assert all(v == pytest.approx(EXPECTED_MEDIAN, rel=1e-6) for v in result_col)

    def test_output_rows_equal_input_rows(self) -> None:
        fs = make_feature_set("value_int__sum_scalar")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert self.get_row_count(result) == 12

    def test_result_has_correct_type(self) -> None:
        fs = make_feature_set("value_int__sum_scalar")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())

    def test_new_column_added(self) -> None:
        fs = make_feature_set("value_int__max_scalar")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__max_scalar")
        assert len(result_col) == 12

    def test_broadcast_uniform(self) -> None:
        """All rows should have the same aggregated value."""
        fs = make_feature_set("value_int__sum_scalar")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__sum_scalar")
        assert len(set(result_col)) == 1

    def test_unsupported_aggregation_type_raises(self) -> None:
        """An unrecognized aggregation type must raise ValueError."""
        fs = make_feature_set("value_int__bogus_scalar")
        with pytest.raises(ValueError, match="[Uu]nsupported|[Cc]ould not extract"):
            self.implementation_class().calculate_feature(self.test_data, fs)

    def test_option_based_single_column_sum(self) -> None:
        """Option-based configuration (not string pattern) produces the same result."""
        feature = Feature(
            "my_custom_sum",
            options=Options(
                context={
                    "aggregation_type": "sum",
                    "in_features": "value_int",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "my_custom_sum")
        assert all(v == EXPECTED_SUM for v in result_col)
        assert len(result_col) == 12

    def test_option_based_single_column_min(self) -> None:
        """Option-based min aggregation produces the correct broadcast value."""
        feature = Feature(
            "my_min",
            options=Options(
                context={
                    "aggregation_type": "min",
                    "in_features": "value_int",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "my_min")
        assert all(v == EXPECTED_MIN for v in result_col)
        assert len(result_col) == 12

    def test_option_based_single_column_max(self) -> None:
        """Option-based max aggregation produces the correct broadcast value."""
        feature = Feature(
            "my_max",
            options=Options(
                context={
                    "aggregation_type": "max",
                    "in_features": "value_int",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "my_max")
        assert all(v == EXPECTED_MAX for v in result_col)
        assert len(result_col) == 12

    def test_multi_column_in_features_rejected_at_calculate(self) -> None:
        """calculate_feature must reject features with multiple in_features.

        This verifies the safety guard in _extract_source_features() that
        prevents silent truncation to a single column.
        """
        feature = Feature(
            "bad_multi",
            options=Options(
                context={
                    "aggregation_type": "sum",
                    "in_features": ["value_int", "value_float"],
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        with pytest.raises(ValueError, match="at most 1"):
            self.implementation_class().calculate_feature(self.test_data, fs)

    # -- Cross-framework comparison ------------------------------------------

    def test_cross_framework_sum(self) -> None:
        self._compare_with_reference("value_int__sum_scalar")

    def test_cross_framework_avg(self) -> None:
        self._compare_with_reference("value_int__avg_scalar", use_approx=True)

    def test_cross_framework_count(self) -> None:
        self._compare_with_reference("value_int__count_scalar")

    def test_cross_framework_min(self) -> None:
        self._compare_with_reference("value_int__min_scalar")

    def test_cross_framework_max(self) -> None:
        self._compare_with_reference("value_int__max_scalar")

    # -- Null consistency tests (all-null column: score) ---------------------

    def test_all_null_column_sum(self) -> None:
        """score is all-null. Sum should broadcast None to every row."""
        fs = make_feature_set("score__sum_scalar")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_col = self.extract_column(result, "score__sum_scalar")
        assert all(v is None for v in result_col)

    def test_all_null_column_min(self) -> None:
        """score is all-null. Min should broadcast None."""
        fs = make_feature_set("score__min_scalar")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_col = self.extract_column(result, "score__min_scalar")
        assert all(v is None for v in result_col)

    def test_all_null_column_max(self) -> None:
        """score is all-null. Max should broadcast None."""
        fs = make_feature_set("score__max_scalar")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_col = self.extract_column(result, "score__max_scalar")
        assert all(v is None for v in result_col)

    def test_all_null_column_avg(self) -> None:
        """score is all-null. Avg should broadcast None."""
        fs = make_feature_set("score__avg_scalar")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_col = self.extract_column(result, "score__avg_scalar")
        assert all(v is None for v in result_col)

    def test_all_null_column_count(self) -> None:
        """score is all-null. Count of non-nulls should broadcast 0."""
        fs = make_feature_set("score__count_scalar")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_col = self.extract_column(result, "score__count_scalar")
        assert all(v == 0 for v in result_col)

    def test_all_null_column_std(self) -> None:
        """score is all-null. Std should broadcast None."""
        self._skip_if_unsupported("std")
        fs = make_feature_set("score__std_scalar")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_col = self.extract_column(result, "score__std_scalar")
        assert all(v is None for v in result_col)

    def test_all_null_column_var(self) -> None:
        """score is all-null. Var should broadcast None."""
        self._skip_if_unsupported("var")
        fs = make_feature_set("score__var_scalar")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_col = self.extract_column(result, "score__var_scalar")
        assert all(v is None for v in result_col)

    def test_all_null_column_median(self) -> None:
        """score is all-null. Median should broadcast None."""
        self._skip_if_unsupported("median")
        fs = make_feature_set("score__median_scalar")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_col = self.extract_column(result, "score__median_scalar")
        assert all(v is None for v in result_col)

    # -- Null consistency tests (multi-null columns) -------------------------

    def test_multi_null_column_count(self) -> None:
        """value_float has 2 nulls (rows 2, 11). Count should be 10."""
        fs = make_feature_set("value_float__count_scalar")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_col = self.extract_column(result, "value_float__count_scalar")
        assert all(v == 10 for v in result_col)

    def test_amount_null_count(self) -> None:
        """amount has 2 nulls (rows 1, 7). Count should be 10."""
        fs = make_feature_set("amount__count_scalar")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_col = self.extract_column(result, "amount__count_scalar")
        assert all(v == 10 for v in result_col)

    # -- Cross-framework null comparisons ------------------------------------

    def test_cross_framework_all_null_min(self) -> None:
        self._compare_with_reference("score__min_scalar")

    def test_cross_framework_all_null_max(self) -> None:
        self._compare_with_reference("score__max_scalar")

    def test_cross_framework_all_null_avg(self) -> None:
        self._compare_with_reference("score__avg_scalar")

    def test_cross_framework_all_null_count(self) -> None:
        self._compare_with_reference("score__count_scalar")

    def test_cross_framework_multi_null_sum(self) -> None:
        self._compare_with_reference("value_float__sum_scalar", use_approx=True)

    def test_cross_framework_multi_null_count(self) -> None:
        self._compare_with_reference("value_float__count_scalar")

    def test_cross_framework_amount_sum(self) -> None:
        self._compare_with_reference("amount__sum_scalar", use_approx=True)

    def test_cross_framework_amount_count(self) -> None:
        self._compare_with_reference("amount__count_scalar")
