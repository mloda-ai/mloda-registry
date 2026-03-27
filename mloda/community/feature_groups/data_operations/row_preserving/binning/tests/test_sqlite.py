"""Tests for SqliteBinning compute implementation."""

from __future__ import annotations

import sqlite3
from typing import Any

import pyarrow as pa

from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.row_preserving.binning.sqlite_binning import (
    SqliteBinning,
)
from mloda.testing.feature_groups.data_operations.row_preserving.binning import (
    BinningTestBase,
)


class TestSqliteBinning(BinningTestBase):
    """All tests inherited from the base class."""

    def setup_method(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        super().setup_method()

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteBinning

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        return SqliteRelation.from_arrow(self.conn, arrow_table)

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        return list(result.to_arrow_table().column(column_name).to_pylist())

    def get_row_count(self, result: Any) -> int:
        return len(result)

    def get_expected_type(self) -> type:
        return SqliteRelation
