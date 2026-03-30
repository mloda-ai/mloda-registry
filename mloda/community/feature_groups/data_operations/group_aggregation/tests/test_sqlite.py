"""Tests for SqliteGroupAggregation compute implementation."""

from __future__ import annotations

import sqlite3
from typing import Any

import pyarrow as pa

from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.group_aggregation.sqlite_group_aggregation import (
    SqliteGroupAggregation,
)
from mloda.testing.feature_groups.data_operations.group_aggregation.group_aggregation import (
    GroupAggregationTestBase,
)


class TestSqliteGroupAggregation(GroupAggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def supported_agg_types(cls) -> set[str]:
        return {"sum", "avg", "count", "min", "max"}

    def setup_method(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        super().setup_method()

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteGroupAggregation

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        return SqliteRelation.from_arrow(self.conn, arrow_table)

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        return list(result.to_arrow_table().column(column_name).to_pylist())

    def get_row_count(self, result: Any) -> int:
        return len(result)

    def get_expected_type(self) -> type:
        return SqliteRelation
