"""Tests for SqliteWindowAggregation compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.sqlite_window_aggregation import (
    SqliteWindowAggregation,
)
from mloda.testing.feature_groups.data_operations.mixins.reserved_columns import ReservedColumnsTestMixin
from mloda.testing.feature_groups.data_operations.mixins.sqlite import SqliteTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation.window_aggregation import (
    WindowAggregationTestBase,
)


class TestSqliteWindowAggregation(ReservedColumnsTestMixin, SqliteTestMixin, WindowAggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def supported_agg_types(cls) -> set[str]:
        return {"sum", "avg", "count", "min", "max"}

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteWindowAggregation

    @classmethod
    def reserved_columns_feature_name(cls) -> str:
        return "value_int__sum_window"

    @classmethod
    def reserved_columns_partition_by(cls) -> list[str] | None:
        return ["region"]

    @classmethod
    def reserved_columns_order_by(cls) -> str | None:
        return None
