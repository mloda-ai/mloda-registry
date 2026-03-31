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

    def test_multiple_null_order_by_values(self) -> None:
        """Two or more null order_by values must not crash during sorting."""
        table = pa.table(
            {
                "region": ["A", "A", "A", "A"],
                "ts": [None, 1, None, 2],
                "value": [100, 10, 200, 20],
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

        # Sorted order: ts=1 (10), ts=2 (20), ts=None (100), ts=None (200)
        # Cumsum:        10,       30,          130,            330
        # Map back to original positions: row0=None->130 or 330, row1=1->10, row2=None->330 or 130, row3=2->30
        assert col[1] == 10
        assert col[3] == 30
        assert set([col[0], col[2]]) == {130, 330}

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
