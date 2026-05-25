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
    def implementation_class(cls) -> Any:
        return SqliteFrameAggregate

    # -- Regression tests for PR #202 review bugs -----------------------------

    def test_sqlite_month_window_end_of_month_matches_reference(self) -> None:
        """Mar 31 -1 month should be Feb 28 (relativedelta), not Mar 3 (SQLite native)."""
        table = pa.table(
            {
                "region": ["A", "A", "A", "A"],
                "ts": [
                    datetime(2023, 2, 28, tzinfo=timezone.utc),
                    datetime(2023, 3, 1, tzinfo=timezone.utc),
                    datetime(2023, 3, 15, tzinfo=timezone.utc),
                    datetime(2023, 3, 31, tzinfo=timezone.utc),
                ],
                "value": [1, 2, 4, 8],
            }
        )
        data = self.create_test_data(table)
        feature_name = "value__sum_1_month_window"
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
