"""Tests for SqliteWindowAggregation compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.sqlite_window_aggregation import (
    SqliteWindowAggregation,
)
from mloda.testing.feature_groups.data_operations.helpers import SqliteTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation import (
    WindowAggregationTestBase,
)


class TestSqliteWindowAggregation(SqliteTestMixin, WindowAggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def supported_agg_types(cls) -> set[str]:
        return {"sum", "avg", "count", "min", "max"}

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteWindowAggregation
