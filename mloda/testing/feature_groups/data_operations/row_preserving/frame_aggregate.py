"""Shared test base class, data, and helpers for frame aggregate tests.

Expected values are computed from the canonical 12-row dataset in
DataOperationsTestDataCreator, partitioned by the 'region' column
(A=rows 0-3, B=rows 4-7, C=rows 8-10, None=row 11) and ordered by
'value_int'.

The ``FrameAggregateTestBase`` class provides concrete test methods that
any framework implementation inherits by subclassing and implementing a
small set of abstract methods. This follows the same pattern as
``WindowAggregationTestBase``.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda.user import Feature


# ---------------------------------------------------------------------------
# Expected values (module-level constants)
# ---------------------------------------------------------------------------

# Rolling sum (window 3) on value_int, partitioned by region, ordered by value_int.
# Group A sorted by value_int: (-5, 0, 10, 20)
#   row1(-5): window=[-5] -> -5
#   row2(0):  window=[-5, 0] -> -5
#   row0(10): window=[-5, 0, 10] -> 5
#   row3(20): window=[0, 10, 20] -> 30
# Group B sorted by value_int: (30, 50, 60, None)
#   row6(30): window=[30] -> 30
#   row5(50): window=[30, 50] -> 80
#   row7(60): window=[30, 50, 60] -> 140
#   row4(None): window=[50, 60, None] -> 110 (null skipped)
# Group C sorted by value_int: (15, 15, 40)
#   row8(15): window=[15] -> 15
#   row9(15): window=[15, 15] -> 30
#   row10(40): window=[15, 15, 40] -> 70
# Group None: (-10)
#   row11(-10): window=[-10] -> -10
#
# Output ordered by original row index: [row0, row1, row2, ..., row11]
EXPECTED_ROLLING_SUM_3: list[int] = [5, -5, -5, 30, 110, 80, 30, 140, 15, 30, 70, -10]

# Cumulative sum on value_int, partitioned by region, ordered by value_int.
# Group A: (-5) -> (-5+0) -> (-5+0+10) -> (-5+0+10+20)
#   = -5, -5, 5, 25
# Group B: (30) -> (30+50) -> (30+50+60) -> (30+50+60+None)
#   = 30, 80, 140, 140
# Group C: (15) -> (15+15) -> (15+15+40) = 15, 30, 70
# Group None: -10
#
# Output ordered by original row index:
EXPECTED_CUMSUM: list[int] = [5, -5, -5, 25, 140, 80, 30, 140, 15, 30, 70, -10]

# Rolling avg (window 2) on value_int, partitioned by region, ordered by value_int.
# Group A sorted: (-5, 0, 10, 20)
#   row1: [-5] -> -5.0
#   row2: [-5, 0] -> -2.5
#   row0: [0, 10] -> 5.0
#   row3: [10, 20] -> 15.0
# Group B sorted: (30, 50, 60, None)
#   row6: [30] -> 30.0
#   row5: [30, 50] -> 40.0
#   row7: [50, 60] -> 55.0
#   row4: [60, None] -> 60.0
# Group C sorted: (15, 15, 40)
#   row8: [15] -> 15.0
#   row9: [15, 15] -> 15.0
#   row10: [15, 40] -> 27.5
# Group None: -10.0
EXPECTED_ROLLING_AVG_2: list[float] = [5.0, -5.0, -2.5, 15.0, 60.0, 40.0, 30.0, 55.0, 15.0, 15.0, 27.5, -10.0]

# Expanding avg on value_int, partitioned by region, ordered by value_int.
# Group A sorted: (-5, 0, 10, 20)
#   row1: [-5] -> -5.0
#   row2: [-5, 0] -> -2.5
#   row0: [-5, 0, 10] -> 1.6667
#   row3: [-5, 0, 10, 20] -> 6.25
# Group B sorted: (30, 50, 60, None)
#   row6: [30] -> 30.0
#   row5: [30, 50] -> 40.0
#   row7: [30, 50, 60] -> 46.6667
#   row4: [30, 50, 60, None] -> 46.6667
# Group C sorted: (15, 15, 40)
#   row8: [15] -> 15.0
#   row9: [15, 15] -> 15.0
#   row10: [15, 15, 40] -> 23.3333
# Group None: -10.0
EXPECTED_EXPANDING_AVG: list[float] = [
    5.0 / 3.0,
    -5.0,
    -2.5,
    6.25,
    140.0 / 3.0,
    40.0,
    30.0,
    140.0 / 3.0,
    15.0,
    15.0,
    70.0 / 3.0,
    -10.0,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_null(value: Any) -> bool:
    """Check if a value is null (None or NaN)."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _assert_values_with_nulls(actual: list[Any], expected: list[Any]) -> None:
    """Assert two lists are equal, treating None and NaN as equivalent nulls."""
    assert len(actual) == len(expected), f"length {len(actual)} != {len(expected)}"
    for i, (a, e) in enumerate(zip(actual, expected)):
        if _is_null(e):
            assert _is_null(a), f"row {i}: expected null, got {a}"
        else:
            assert a == pytest.approx(e, rel=1e-6), f"row {i}: {a} != {e}"


# ---------------------------------------------------------------------------
# Reusable test base class
# ---------------------------------------------------------------------------


class FrameAggregateTestBase(DataOpsTestBase):
    """Abstract base class for frame aggregate framework tests."""

    @classmethod
    def pyarrow_implementation_class(cls) -> Any:
        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.pyarrow_frame_aggregate import (
            PyArrowFrameAggregate,
        )

        return PyArrowFrameAggregate

    @classmethod
    def supports_time_frame(cls) -> bool:
        """Whether this framework supports frame_type='time'. Default: False."""
        return False

    # -- Rolling tests -------------------------------------------------------

    def test_rolling_sum_window_3(self) -> None:
        """Rolling sum with window size 3, partitioned by region, ordered by value_int."""
        fs = make_feature_set("value_int__sum_rolling_3", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__sum_rolling_3")
        assert result_col == EXPECTED_ROLLING_SUM_3

    def test_rolling_avg_window_2(self) -> None:
        """Rolling avg with window size 2, partitioned by region, ordered by value_int."""
        fs = make_feature_set("value_int__avg_rolling_2", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__avg_rolling_2")
        assert result_col == pytest.approx(EXPECTED_ROLLING_AVG_2, rel=1e-3)

    # -- Cumulative tests ----------------------------------------------------

    def test_cumulative_sum(self) -> None:
        """Cumulative sum, partitioned by region, ordered by value_int."""
        fs = make_feature_set("value_int__cumsum", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__cumsum")
        assert result_col == EXPECTED_CUMSUM

    # -- Expanding tests -----------------------------------------------------

    def test_expanding_avg(self) -> None:
        """Expanding avg, partitioned by region, ordered by value_int."""
        fs = make_feature_set("value_int__expanding_avg", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__expanding_avg")
        assert result_col == pytest.approx(EXPECTED_EXPANDING_AVG, rel=1e-3)

    # -- Row preservation ----------------------------------------------------

    def test_output_rows_equal_input_rows(self) -> None:
        """Output must have exactly 12 rows, same as input."""
        fs = make_feature_set("value_int__sum_rolling_3", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert self.get_row_count(result) == 12

    def test_result_has_correct_type(self) -> None:
        """The result of calculate_feature must be the expected framework type."""
        fs = make_feature_set("value_int__cumsum", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())

    # -- Time window support tests -------------------------------------------

    def test_time_frame_match_rejected_when_unsupported(self) -> None:
        """Frameworks that do not support time frames must reject them at match time."""
        if self.supports_time_frame():
            pytest.skip("This framework supports time frames")

        options = Options(context={"partition_by": ["region"], "order_by": "timestamp"})
        assert not self.implementation_class().match_feature_group_criteria("value_int__avg_7_day_window", options)

    def test_time_frame_config_rejected_when_unsupported(self) -> None:
        """Config-based time frame features must be rejected at match time when unsupported."""
        if self.supports_time_frame():
            pytest.skip("This framework supports time frames")

        options = Options(
            context={
                "aggregation_type": "sum",
                "frame_type": "time",
                "frame_size": 2,
                "frame_unit": "day",
                "partition_by": ["region"],
                "order_by": "timestamp",
            }
        )
        assert not self.implementation_class().match_feature_group_criteria("my_time_sum", options)

    # -- Calendar-accurate time window tests ----------------------------------

    def test_month_window_handles_variable_length_months(self) -> None:
        """A 1-month window should use calendar months, not a fixed 30-day offset."""
        if not self.supports_time_frame():
            pytest.skip("This framework does not support time frames")

        table = pa.table(
            {
                "region": ["A", "A", "A"],
                "ts": [
                    datetime(2023, 1, 1, tzinfo=timezone.utc),
                    datetime(2023, 1, 31, tzinfo=timezone.utc),
                    datetime(2023, 3, 1, tzinfo=timezone.utc),
                ],
                "value": [10, 20, 30],
            }
        )
        data = self.create_test_data(table)
        feature = Feature(
            "monthly_sum",
            options=Options(
                context={
                    "aggregation_type": "sum",
                    "frame_type": "time",
                    "frame_size": 1,
                    "frame_unit": "month",
                    "in_features": "value",
                    "partition_by": ["region"],
                    "order_by": "ts",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = self.implementation_class().calculate_feature(data, fs)
        col = self.extract_column(result, "monthly_sum")

        # Row 0 (Jan 1): window [Jan 1] => 10
        # Row 1 (Jan 31): 1 month back = Dec 31. Jan 1 >= Dec 31, so window [10, 20] => 30
        # Row 2 (Mar 1): 1 month back = Feb 1. Jan 31 < Feb 1, so window [30] => 30
        assert col == [10, 30, 30]

    def test_year_window_handles_leap_year(self) -> None:
        """A 1-year window should use calendar years, not a fixed 365-day offset."""
        if not self.supports_time_frame():
            pytest.skip("This framework does not support time frames")

        table = pa.table(
            {
                "region": ["A", "A", "A"],
                "ts": [
                    datetime(2024, 1, 1, tzinfo=timezone.utc),
                    datetime(2024, 12, 31, tzinfo=timezone.utc),
                    datetime(2025, 1, 1, tzinfo=timezone.utc),
                ],
                "value": [10, 20, 30],
            }
        )
        data = self.create_test_data(table)
        feature = Feature(
            "yearly_sum",
            options=Options(
                context={
                    "aggregation_type": "sum",
                    "frame_type": "time",
                    "frame_size": 1,
                    "frame_unit": "year",
                    "in_features": "value",
                    "partition_by": ["region"],
                    "order_by": "ts",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = self.implementation_class().calculate_feature(data, fs)
        col = self.extract_column(result, "yearly_sum")

        # Row 0 (2024-01-01): window [10] => 10
        # Row 1 (2024-12-31): 1 year back = 2023-12-31. 2024-01-01 >= 2023-12-31, so [10, 20] => 30
        # Row 2 (2025-01-01): 1 year back = 2024-01-01. 2024-01-01 >= 2024-01-01, so [10, 20, 30] => 60
        assert col == [10, 30, 60]

    # -- Cross-framework comparison ------------------------------------------

    def test_cross_framework_rolling_sum(self) -> None:
        """Rolling sum must match PyArrow reference."""
        self._compare_with_pyarrow("value_int__sum_rolling_3", partition_by=["region"], order_by="value_int")

    def test_cross_framework_cumsum(self) -> None:
        """Cumulative sum must match PyArrow reference."""
        self._compare_with_pyarrow("value_int__cumsum", partition_by=["region"], order_by="value_int")

    def test_cross_framework_expanding_avg(self) -> None:
        """Expanding avg must match PyArrow reference."""
        self._compare_with_pyarrow(
            "value_int__expanding_avg", partition_by=["region"], order_by="value_int", use_approx=True
        )

    def test_cross_framework_rolling_avg(self) -> None:
        """Rolling avg must match PyArrow reference."""
        self._compare_with_pyarrow(
            "value_int__avg_rolling_2", partition_by=["region"], order_by="value_int", use_approx=True
        )

    # -- Edge case tests -----------------------------------------------------

    def test_rolling_window_size_1(self) -> None:
        """Rolling with window_size=1 should return the value itself."""
        fs = make_feature_set("value_int__sum_rolling_1", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__sum_rolling_1")
        # Window of 1 means each value is its own window, so the sum equals the original value.
        # Original row order: [10, -5, 0, 20, None, 50, 30, 60, 15, 15, 40, -10]
        # Nulls in sum_rolling_1 should produce None (or NaN in Pandas).
        expected = [10, -5, 0, 20, None, 50, 30, 60, 15, 15, 40, -10]
        _assert_values_with_nulls(result_col, expected)

    def test_all_null_values_returns_null(self) -> None:
        """When all values in the source column are null, results should be null."""
        all_null_table = pa.table(
            {
                "region": ["A", "A", "A"],
                "value_int": [None, None, None],
            }
        )
        data = self.create_test_data(all_null_table)
        fs = make_feature_set("value_int__cumsum", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(data, fs)

        assert self.get_row_count(result) == 3
        result_col = self.extract_column(result, "value_int__cumsum")
        for i, val in enumerate(result_col):
            assert _is_null(val), f"row {i}: expected null, got {val}"

    def test_window_larger_than_partition(self) -> None:
        """Rolling window larger than partition should include all rows in the partition."""
        fs = make_feature_set("value_int__sum_rolling_100", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__sum_rolling_100")
        # With window 100, each row sees all preceding rows in its partition.
        # This is equivalent to cumulative sum.
        assert result_col == EXPECTED_CUMSUM

    def test_multiple_null_order_by_values(self) -> None:
        """Two or more null order_by values must not crash during sorting."""
        table = pa.table(
            {
                "region": ["A", "A", "A", "A"],
                "ts": [None, 1, None, 2],
                "value": [100, 10, 200, 20],
            }
        )
        data = self.create_test_data(table)
        fs = make_feature_set("value__cumsum", ["region"], "ts")
        result = self.implementation_class().calculate_feature(data, fs)

        result_col = self.extract_column(result, "value__cumsum")

        # Sorted order: ts=1 (10), ts=2 (20), ts=None (100), ts=None (200)
        # Cumsum:        10,       30,          130,            330
        # Map back to original positions: row1->10, row3->30, nulls get 130 and 330
        assert result_col[1] == 10
        assert result_col[3] == 30
        assert set([result_col[0], result_col[2]]) == {130, 330}
