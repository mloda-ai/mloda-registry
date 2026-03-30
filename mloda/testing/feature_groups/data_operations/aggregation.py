"""Shared test base class and expected values for single-column aggregate broadcast tests.

Each test verifies that a scalar aggregate (sum, min, max, etc.) is computed
over a single source column and the result is broadcast to every row.

Expected values are computed from the canonical 12-row dataset in
DataOperationsTestDataCreator, using the 'value_int' column.

value_int = [10, -5, 0, 20, None, 50, 30, 60, 15, 15, 40, -10]
Non-null values: [10, -5, 0, 20, 50, 30, 60, 15, 15, 40, -10] (11 values)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
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

# Median of sorted non-null: [-10, -5, 0, 10, 15, 15, 20, 30, 40, 50, 60] -> 15.0
EXPECTED_MEDIAN: float = 15.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_column(result: Any, column_name: str) -> list[Any]:
    """Extract a column from a result object as a Python list."""
    if isinstance(result, pa.Table):
        return list(result.column(column_name).to_pylist())
    arrow_table = result.to_arrow_table()
    return list(arrow_table.column(column_name).to_pylist())


def make_feature_set(feature_name: str) -> FeatureSet:
    """Build a FeatureSet for an aggregation feature."""
    feature = Feature(feature_name, options=Options())
    fs = FeatureSet()
    fs.add(feature)
    return fs


# ---------------------------------------------------------------------------
# Reusable test base class
# ---------------------------------------------------------------------------


class AggregationTestBase(ABC):
    """Abstract base class for column aggregation framework tests."""

    ALL_AGG_TYPES = {"sum", "min", "max", "avg", "mean", "count", "std", "var", "median"}

    @classmethod
    def supported_agg_types(cls) -> set[str]:
        """Override to restrict supported types for a framework."""
        return cls.ALL_AGG_TYPES

    @classmethod
    @abstractmethod
    def implementation_class(cls) -> Any:
        """Return the aggregation implementation class to test."""

    @classmethod
    def pyarrow_implementation_class(cls) -> Any:
        from mloda.community.feature_groups.data_operations.aggregation.pyarrow_aggregation import (
            PyArrowColumnAggregation,
        )

        return PyArrowColumnAggregation

    @abstractmethod
    def create_test_data(self, arrow_table: pa.Table) -> Any:
        """Convert PyArrow table to framework's native format."""

    @abstractmethod
    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        """Extract a column as a Python list."""

    @abstractmethod
    def get_row_count(self, result: Any) -> int:
        """Return row count."""

    @abstractmethod
    def get_expected_type(self) -> Any:
        """Return expected result type."""

    # -- Setup / teardown ----------------------------------------------------

    def setup_method(self) -> None:
        self._arrow_table = PyArrowDataOpsTestDataCreator.create()
        self.test_data = self.create_test_data(self._arrow_table)

    def teardown_method(self) -> None:
        conn = getattr(self, "conn", None)
        if conn is not None:
            conn.close()

    # -- Helpers -------------------------------------------------------------

    def _skip_if_unsupported(self, agg_type: str) -> None:
        if agg_type not in self.supported_agg_types():
            pytest.skip(f"{agg_type} not supported by this framework")

    # -- Concrete tests ------------------------------------------------------

    def test_sum_aggr(self) -> None:
        """Sum of value_int broadcast to all rows."""
        fs = make_feature_set("value_int__sum_aggr")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__sum_aggr")
        assert all(v == EXPECTED_SUM for v in result_col)

    def test_min_aggr(self) -> None:
        fs = make_feature_set("value_int__min_aggr")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__min_aggr")
        assert all(v == EXPECTED_MIN for v in result_col)

    def test_max_aggr(self) -> None:
        fs = make_feature_set("value_int__max_aggr")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__max_aggr")
        assert all(v == EXPECTED_MAX for v in result_col)

    def test_avg_aggr(self) -> None:
        fs = make_feature_set("value_int__avg_aggr")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__avg_aggr")
        assert all(v == pytest.approx(EXPECTED_AVG, rel=1e-6) for v in result_col)

    def test_count_aggr(self) -> None:
        fs = make_feature_set("value_int__count_aggr")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__count_aggr")
        assert all(v == EXPECTED_COUNT for v in result_col)

    def test_std_aggr(self) -> None:
        self._skip_if_unsupported("std")
        fs = make_feature_set("value_int__std_aggr")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__std_aggr")
        assert all(v == pytest.approx(EXPECTED_STD, rel=1e-4) for v in result_col)

    def test_var_aggr(self) -> None:
        self._skip_if_unsupported("var")
        fs = make_feature_set("value_int__var_aggr")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__var_aggr")
        assert all(v == pytest.approx(EXPECTED_VAR, rel=1e-4) for v in result_col)

    def test_median_aggr(self) -> None:
        self._skip_if_unsupported("median")
        fs = make_feature_set("value_int__median_aggr")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__median_aggr")
        assert all(v == pytest.approx(EXPECTED_MEDIAN, rel=1e-6) for v in result_col)

    def test_output_rows_equal_input_rows(self) -> None:
        fs = make_feature_set("value_int__sum_aggr")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert self.get_row_count(result) == 12

    def test_result_has_correct_type(self) -> None:
        fs = make_feature_set("value_int__sum_aggr")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())

    def test_new_column_added(self) -> None:
        fs = make_feature_set("value_int__max_aggr")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__max_aggr")
        assert len(result_col) == 12

    def test_broadcast_uniform(self) -> None:
        """All rows should have the same aggregated value."""
        fs = make_feature_set("value_int__sum_aggr")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__sum_aggr")
        assert len(set(result_col)) == 1

    def test_unsupported_aggregation_type_raises(self) -> None:
        """An unrecognized aggregation type must raise ValueError."""
        fs = make_feature_set("value_int__bogus_aggr")
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

    def _compare_with_pyarrow(self, feature_name: str, use_approx: bool = False) -> None:
        fs = make_feature_set(feature_name)
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        ref = self.pyarrow_implementation_class().calculate_feature(self._arrow_table, fs)

        result_col = self.extract_column(result, feature_name)
        ref_col = extract_column(ref, feature_name)

        assert len(result_col) == len(ref_col)
        if use_approx:
            for i, (r, f) in enumerate(zip(ref_col, result_col)):
                if r is None:
                    assert f is None, f"row {i}: expected None, got {f}"
                else:
                    assert f == pytest.approx(r, rel=1e-4), f"row {i}: {f} != reference {r}"
        else:
            assert result_col == ref_col

    def test_cross_framework_sum(self) -> None:
        self._compare_with_pyarrow("value_int__sum_aggr")

    def test_cross_framework_avg(self) -> None:
        self._compare_with_pyarrow("value_int__avg_aggr", use_approx=True)

    def test_cross_framework_count(self) -> None:
        self._compare_with_pyarrow("value_int__count_aggr")

    def test_cross_framework_min(self) -> None:
        self._compare_with_pyarrow("value_int__min_aggr")

    def test_cross_framework_max(self) -> None:
        self._compare_with_pyarrow("value_int__max_aggr")
