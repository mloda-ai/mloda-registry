"""Tests for PyArrow frame aggregate implementation.

Uses the unified FrameAggregateTestBase plus PyArrow-specific tests
for time windows and null handling.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.row_preserving.frame_aggregate import (
    FrameAggregateTestBase,
)
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.pyarrow_frame_aggregate import (
    PyArrowFrameAggregate,
)


class TestPyArrowFrameAggregate(FrameAggregateTestBase):
    """Unified tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowFrameAggregate

    @classmethod
    def supports_time_frame(cls) -> bool:
        return True

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        return arrow_table

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        return list(result.column(column_name).to_pylist())

    def get_row_count(self, result: Any) -> int:
        return int(result.num_rows)

    def get_expected_type(self) -> Any:
        return pa.Table


class TestPyArrowTimeWindow:
    """PyArrow-specific time window tests (only PyArrow supports time frames)."""

    def test_time_frame_match_accepted(self) -> None:
        """PyArrow supports time frames, so matching should succeed."""
        options = Options(context={"partition_by": ["region"], "order_by": "timestamp"})
        assert PyArrowFrameAggregate.match_feature_group_criteria("value_int__avg_7_day_window", options)

    def test_time_window_via_config(self) -> None:
        """Time-based window using config-based feature creation."""
        table = pa.table(
            {
                "region": ["A", "A", "A", "A"],
                "ts": [
                    datetime(2023, 1, 1, tzinfo=timezone.utc),
                    datetime(2023, 1, 2, tzinfo=timezone.utc),
                    datetime(2023, 1, 4, tzinfo=timezone.utc),
                    datetime(2023, 1, 5, tzinfo=timezone.utc),
                ],
                "value": [10, 20, 30, 40],
            }
        )
        feature = Feature(
            "my_time_sum",
            options=Options(
                context={
                    "aggregation_type": "sum",
                    "frame_type": "time",
                    "frame_size": 2,
                    "frame_unit": "day",
                    "in_features": "value",
                    "partition_by": ["region"],
                    "order_by": "ts",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = PyArrowFrameAggregate.calculate_feature(table, fs)
        col = result.column("my_time_sum").to_pylist()

        # 2-day window:
        # Jan 1: window includes Jan 1 (within 2 days back) => [10] => 10
        # Jan 2: window includes Jan 1, Jan 2 => [10, 20] => 30
        # Jan 4: window includes Jan 2, Jan 4 (Jan 1 is >2 days before Jan 4) => [20, 30] => 50
        # Jan 5: window includes Jan 4, Jan 5 (Jan 2 is >2 days before Jan 5) => [30, 40] => 70
        assert col == [10, 30, 50, 70]


class TestPyArrowNullHandling:
    """PyArrow-specific null handling edge case tests."""

    def test_rolling_sum_skips_nulls(self) -> None:
        """Null values in the source column should be skipped by the aggregation."""
        table = pa.table(
            {
                "region": ["A", "A", "A", "A"],
                "ts": [1, 2, 3, 4],
                "value": [10, None, 30, 40],
            }
        )
        feature = Feature(
            "value__sum_rolling_3",
            options=Options(context={"partition_by": ["region"], "order_by": "ts"}),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = PyArrowFrameAggregate.calculate_feature(table, fs)
        col = result.column("value__sum_rolling_3").to_pylist()

        assert col == [10, 10, 40, 70]

    def test_cumsum_with_null_order_by(self) -> None:
        """Rows with null order_by values should be sorted last."""
        table = pa.table(
            {
                "region": ["A", "A", "A"],
                "ts": [None, 1, 2],
                "value": [100, 10, 20],
            }
        )
        feature = Feature(
            "value__cumsum",
            options=Options(context={"partition_by": ["region"], "order_by": "ts"}),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = PyArrowFrameAggregate.calculate_feature(table, fs)
        col = result.column("value__cumsum").to_pylist()

        assert col == [130, 10, 30]

    def test_all_null_values_returns_none(self) -> None:
        """When all values in the window are null, the result should be None."""
        table = pa.table(
            {
                "region": ["A", "A"],
                "ts": [1, 2],
                "value": [None, None],
            }
        )
        feature = Feature(
            "value__cumsum",
            options=Options(context={"partition_by": ["region"], "order_by": "ts"}),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = PyArrowFrameAggregate.calculate_feature(table, fs)
        col = result.column("value__cumsum").to_pylist()

        assert col == [None, None]


class TestPyArrowCalendarAccurateTimeWindow:
    """Tests that time windows for month and year units are calendar-accurate.

    Month and year windows use dateutil.relativedelta for proper calendar
    arithmetic, correctly handling variable-length months (28-31 days) and
    leap years (366 days).
    """

    def test_month_window_handles_variable_length_months(self) -> None:
        """A 1-month window should use calendar months, not a fixed 30-day offset.

        Setup: partition "A" with dates Jan 1, Jan 31, Mar 1 and values 10, 20, 30.
        For a 1-month sum window ending at Mar 1:
        - Calendar-accurate: 1 month before Mar 1 is Feb 1.
          Jan 31 < Feb 1, so Jan 31 is outside the window. Sum = 30.
        - 30-day approximation: 30 days before Mar 1 is Jan 30.
          Jan 31 >= Jan 30, so Jan 31 is inside the window. Sum = 50.

        The buggy code returns 50 at Mar 1 because it includes Jan 31
        (which is 29 days before Mar 1, within the 30-day window). The
        correct answer is 30 because Jan 31 is not within 1 calendar
        month of Mar 1.
        """
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

        result = PyArrowFrameAggregate.calculate_feature(table, fs)
        col = result.column("monthly_sum").to_pylist()

        # Row 0 (Jan 1): window [Jan 1] => 10
        # Row 1 (Jan 31): 1 month back = Dec 31. Jan 1 >= Dec 31, so window [10, 20] => 30
        # Row 2 (Mar 1): 1 month back = Feb 1. Jan 31 < Feb 1, so window [30] => 30
        assert col == [10, 30, 30]

    def test_year_window_handles_leap_year(self) -> None:
        """A 1-year window should use calendar years, not a fixed 365-day offset.

        Setup: dates 2024-01-01, 2024-12-31, 2025-01-01 with values 10, 20, 30.
        2024 is a leap year (366 days).

        For a 1-year sum window ending at 2025-01-01:
        - Calendar-accurate: 1 year before 2025-01-01 is 2024-01-01.
          2024-01-01 >= 2024-01-01, so all three dates are inside. Sum = 60.
        - 365-day approximation: 365 days before 2025-01-01 is 2024-01-02.
          2024-01-01 < 2024-01-02, so 2024-01-01 is excluded. Sum = 50.

        The buggy code returns 50 at 2025-01-01 because the leap year has
        366 days but the code only subtracts 365. The correct answer is 60
        because 2024-01-01 is exactly 1 calendar year before 2025-01-01.
        """
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

        result = PyArrowFrameAggregate.calculate_feature(table, fs)
        col = result.column("yearly_sum").to_pylist()

        # Row 0 (2024-01-01): window [10] => 10
        # Row 1 (2024-12-31): 1 year back = 2023-12-31. 2024-01-01 >= 2023-12-31, so [10, 20] => 30
        # Row 2 (2025-01-01): 1 year back = 2024-01-01. 2024-01-01 >= 2024-01-01, so [10, 20, 30] => 60
        assert col == [10, 30, 60]
