"""Tests for the reference frame aggregate implementation.

Covers time window and null handling edge cases using
the ReferenceFrameAggregate from the testing library.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pyarrow as pa

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.user import Feature

from mloda.testing.feature_groups.data_operations.row_preserving.frame_aggregate.reference import (
    ReferenceFrameAggregate,
)


class TestReferenceTimeWindow:
    """Reference-specific time window tests (only reference supports time frames)."""

    def test_time_frame_match_accepted(self) -> None:
        """Reference supports time frames, so matching should succeed."""
        options = Options(context={"partition_by": ["region"], "order_by": "timestamp"})
        assert ReferenceFrameAggregate.match_feature_group_criteria("value_int__avg_7_day_window", options)

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

        result = ReferenceFrameAggregate.calculate_feature(table, fs)
        col = result.column("my_time_sum").to_pylist()

        # 2-day window:
        # Jan 1: window includes Jan 1 (within 2 days back) => [10] => 10
        # Jan 2: window includes Jan 1, Jan 2 => [10, 20] => 30
        # Jan 4: window includes Jan 2, Jan 4 (Jan 1 is >2 days before Jan 4) => [20, 30] => 50
        # Jan 5: window includes Jan 4, Jan 5 (Jan 2 is >2 days before Jan 5) => [30, 40] => 70
        assert col == [10, 30, 50, 70]


class TestReferenceNullHandling:
    """Reference-specific null handling edge case tests."""

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

        result = ReferenceFrameAggregate.calculate_feature(table, fs)
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

        result = ReferenceFrameAggregate.calculate_feature(table, fs)
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

        result = ReferenceFrameAggregate.calculate_feature(table, fs)
        col = result.column("value__cumsum").to_pylist()

        assert col == [None, None]
