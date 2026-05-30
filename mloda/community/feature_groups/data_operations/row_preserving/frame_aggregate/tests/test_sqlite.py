"""Tests for SQLite frame aggregate implementation.

Uses the unified FrameAggregateTestBase.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pyarrow as pa

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.helpers import extract_column as _extract_column
from mloda.testing.feature_groups.data_operations.mixins.sqlite import SqliteTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.frame_aggregate.frame_aggregate import (
    FrameAggregateTestBase,
)
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.sqlite_frame_aggregate import (
    SqliteFrameAggregate,
)


class TestSqliteFrameAggregate(SqliteTestMixin, FrameAggregateTestBase):
    """Unified tests inherited from the base class."""

    @classmethod
    def reserved_columns_enforced(cls) -> bool:
        return False

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteFrameAggregate

    @classmethod
    def supported_time_units(cls) -> set[str]:
        # SQLite's native ``datetime(ts, '-N months')`` rolls over by day-of-month
        # (Mar 31 -1mo = Mar 3) whereas the reference uses ``relativedelta``
        # (= Feb 28). Rather than fall back to a Python loop (which would defeat
        # the point of running inside the SQLite engine), we reject month/year
        # at match time. See known-divergences.md.
        return {"second", "minute", "hour", "day", "week"}

    # -- Regression tests for PR #202 review bugs -----------------------------

    def test_sqlite_rejects_month_time_window_at_match_time(self) -> None:
        """SQLite rejects month/year time windows because its calendar arithmetic diverges from the reference."""
        options = Options(context={"partition_by": ["region"], "order_by": "ts"})
        assert not self.implementation_class().match_feature_group_criteria("value__sum_1_month_window", options)
        assert not self.implementation_class().match_feature_group_criteria("value__sum_1_year_window", options)
        # day/week still match
        assert self.implementation_class().match_feature_group_criteria("value__sum_3_day_window", options)
        assert self.implementation_class().match_feature_group_criteria("value__sum_1_week_window", options)

    def test_sqlite_second_window_preserves_subsecond_precision(self) -> None:
        """3-second window should respect sub-second precision.

        Row 0 at 12:00:00.000 is 3.5s before row 1 at 12:00:03.500 and must be excluded.
        """
        table = pa.table(
            {
                "region": ["A", "A"],
                "ts": [
                    datetime(2023, 1, 1, 12, 0, 0, 0, tzinfo=timezone.utc),
                    datetime(2023, 1, 1, 12, 0, 3, 500_000, tzinfo=timezone.utc),
                ],
                "value": [1, 2],
            }
        )
        data = self.create_test_data(table)
        feature_name = "value__sum_3_second_window"
        feature = Feature(
            feature_name,
            options=Options(context={"partition_by": ["region"], "order_by": "ts"}),
        )
        fs = FeatureSet()
        fs.add(feature)
        result = self.implementation_class().calculate_feature(data, fs)
        ref = self.reference_implementation_class().calculate_feature(table, fs)
        result_col = self.extract_column(result, feature_name)
        ref_col = _extract_column(ref, feature_name)
        assert result_col == ref_col, f"got {result_col}, expected {ref_col}"
