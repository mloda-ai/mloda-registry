"""Tests for the reference frame aggregate implementation.

Covers time window edge cases (null handling lives in the frame pinning rows) using
reference_frame_aggregate from the testing library.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

import pyarrow as pa
import pytest
from mloda.provider import FeatureSet
from mloda.user import Feature, Options

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.base import (
    FrameAggregateFeatureGroup,
)
from mloda.testing.feature_groups.data_operations.row_preserving.frame_aggregate.reference import (
    reference_frame_aggregate,
)


class TestReferenceTimeWindow:
    """Reference-specific time window tests (only reference supports time frames)."""

    def test_time_frame_match_accepted(self) -> None:
        """Reference supports time frames, so matching should succeed."""
        options = Options(context={"partition_by": ["region"], "order_by": "timestamp"})
        assert FrameAggregateFeatureGroup.match_feature_group_criteria("value_int__avg_7_day_window", options)

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
        col = self._time_sum(table, 2, "day")

        # 2-day window:
        # Jan 1: window includes Jan 1 (within 2 days back) => [10] => 10
        # Jan 2: window includes Jan 1, Jan 2 => [10, 20] => 30
        # Jan 4: window includes Jan 2, Jan 4 (Jan 1 is >2 days before Jan 4) => [20, 30] => 50
        # Jan 5: window includes Jan 4, Jan 5 (Jan 2 is >2 days before Jan 5) => [30, 40] => 70
        assert col == [10, 30, 50, 70]

    @staticmethod
    def _time_sum(table: pa.Table, size: int, unit: str) -> list[int]:
        """Sum of `value` over a time frame of `size` units, ordered by `ts` within `region`."""
        feature = Feature(
            "my_time_sum",
            options=Options(
                context={
                    "aggregation_type": "sum",
                    "frame_type": "time",
                    "frame_size": size,
                    "frame_unit": unit,
                    "in_features": "value",
                    "partition_by": ["region"],
                    "order_by": "ts",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        values: list[int] = (
            reference_frame_aggregate(FrameAggregateFeatureGroup, table, fs).column("my_time_sum").to_pylist()
        )
        return values

    @staticmethod
    def _calendar_sum(stamps: list[datetime], unit: str) -> list[int]:
        """Sum over a 1-unit time frame; row i holds the value 10**i, so each sum names its members."""
        table = pa.table({"region": ["A"] * len(stamps), "ts": stamps, "value": [10**i for i in range(len(stamps))]})
        return TestReferenceTimeWindow._time_sum(table, 1, unit)

    @pytest.mark.parametrize(
        "stamps,unit",
        [
            pytest.param(
                [datetime(2023, 2, d, tzinfo=timezone.utc) for d in (27, 28)]
                + [datetime(2023, 3, 31, tzinfo=timezone.utc)],
                "month",
                id="month_clamps_to_february_end_non_leap_year",
            ),
            pytest.param(
                [datetime(2024, 2, d, tzinfo=timezone.utc) for d in (28, 29)]
                + [datetime(2024, 3, 31, tzinfo=timezone.utc)],
                "month",
                id="month_clamps_to_february_end_leap_year",
            ),
            pytest.param(
                [datetime(2023, 2, d, tzinfo=timezone.utc) for d in (27, 28)]
                + [datetime(2024, 2, 29, tzinfo=timezone.utc)],
                "year",
                id="year_from_leap_day_reaches_february_28",
            ),
            pytest.param(
                [
                    datetime(2023, 2, 28, 11, tzinfo=timezone.utc),
                    datetime(2023, 2, 28, 13, tzinfo=timezone.utc),
                    datetime(2023, 3, 31, 12, tzinfo=timezone.utc),
                ],
                "month",
                id="month_start_keeps_the_time_of_day",
            ),
        ],
    )
    def test_calendar_window_start_keeps_the_last_two_rows(self, stamps: list[datetime], unit: str) -> None:
        """Window start is the last stamp minus one unit, day clamped: the 2nd row is in, the 1st is out."""
        assert self._calendar_sum(stamps, unit)[-1] == 110

    def test_month_window_runs_without_dateutil(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "dateutil", None)
        monkeypatch.setitem(sys.modules, "dateutil.relativedelta", None)
        stamps = [datetime(2023, 2, 28, tzinfo=timezone.utc), datetime(2023, 3, 31, tzinfo=timezone.utc)]
        assert self._calendar_sum(stamps, "month") == [1, 11]
