"""Tests for the reference frame aggregate implementation.

Covers time window and null handling edge cases using
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

    def test_month_window_runs_without_dateutil(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "dateutil", None)
        monkeypatch.setitem(sys.modules, "dateutil.relativedelta", None)
        stamps = [datetime(2023, 2, 28, tzinfo=timezone.utc), datetime(2023, 3, 31, tzinfo=timezone.utc)]
        assert self._calendar_sum(stamps, "month") == [1, 11]


class TestReferenceNullHandling:
    """Reference-specific null handling edge case tests."""

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

        result = reference_frame_aggregate(FrameAggregateFeatureGroup, table, fs)
        col = result.column("value__cumsum").to_pylist()

        assert col == [None, None]
