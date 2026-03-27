"""Tests for SQLite frame aggregate implementation.

Uses the unified FrameAggregateTestBase.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pyarrow as pa
import pytest

from mloda.testing.feature_groups.data_operations.row_preserving.frame_aggregate import (
    FrameAggregateTestBase,
)
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.sqlite_frame_aggregate import (
    SqliteFrameAggregate,
)


class TestSqliteFrameAggregate(FrameAggregateTestBase):
    """Unified tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteFrameAggregate

    def setup_method(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        super().setup_method()

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        return SqliteRelation.from_arrow(self.conn, arrow_table)

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        return list(result.to_arrow_table().column(column_name).to_pylist())

    def get_row_count(self, result: Any) -> int:
        return len(result)

    def get_expected_type(self) -> Any:
        return SqliteRelation
