"""Tests for SqliteFilteredAggregation compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.filtered_aggregation.sqlite_filtered_aggregation import (
    SqliteFilteredAggregation,
)
from mloda.testing.feature_groups.data_operations.mixins.sqlite import SqliteTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.filtered_aggregation.filtered_aggregation import (
    FilteredAggregationTestBase,
)


class TestSqliteFilteredAggregation(SqliteTestMixin, FilteredAggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def supported_agg_types(cls) -> set[str]:
        return {"sum", "avg", "count", "min", "max"}

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteFilteredAggregation
