"""Tests for SqliteGroupAggregation compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.group_aggregation.sqlite_group_aggregation import (
    SqliteGroupAggregation,
)
from mloda.testing.feature_groups.data_operations.group_aggregation.group_aggregation import (
    GroupAggregationTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.sqlite import SqliteTestMixin


class TestSqliteGroupAggregation(SqliteTestMixin, GroupAggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def supported_agg_types(cls) -> set[str]:
        return {"sum", "avg", "count", "min", "max"}

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteGroupAggregation
