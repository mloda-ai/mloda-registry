"""Tests for SqliteColumnAggregation compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.aggregation.sqlite_aggregation import (
    SqliteColumnAggregation,
)
from mloda.testing.feature_groups.data_operations.aggregation.aggregation import (
    AggregationTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.sqlite import SqliteTestMixin


class TestSqliteColumnAggregation(SqliteTestMixin, AggregationTestBase):
    @classmethod
    def supported_agg_types(cls) -> set[str]:
        return {"sum", "min", "max", "avg", "mean", "count"}

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteColumnAggregation
