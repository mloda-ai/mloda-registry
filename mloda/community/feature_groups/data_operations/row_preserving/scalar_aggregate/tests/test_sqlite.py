"""Tests for SqliteScalarAggregate compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.sqlite_scalar_aggregate import (
    SqliteScalarAggregate,
)
from mloda.testing.feature_groups.data_operations.row_preserving.scalar_aggregate.scalar_aggregate import (
    ScalarAggregateTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.sqlite import SqliteTestMixin


class TestSqliteScalarAggregate(SqliteTestMixin, ScalarAggregateTestBase):
    @classmethod
    def supported_agg_types(cls) -> set[str]:
        return {"sum", "min", "max", "avg", "mean", "count"}

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteScalarAggregate
