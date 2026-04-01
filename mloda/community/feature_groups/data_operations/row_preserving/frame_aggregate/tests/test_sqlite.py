"""Tests for SQLite frame aggregate implementation.

Uses the unified FrameAggregateTestBase.
"""

from __future__ import annotations

from typing import Any

from mloda.testing.feature_groups.data_operations.helpers import SqliteTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.frame_aggregate import (
    FrameAggregateTestBase,
)

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.sqlite_frame_aggregate import (
    SqliteFrameAggregate,
)


class TestSqliteFrameAggregate(SqliteTestMixin, FrameAggregateTestBase):
    """Unified tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteFrameAggregate
