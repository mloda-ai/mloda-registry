"""Tests for SqliteOffset compute implementation."""

from __future__ import annotations

import sqlite3
from typing import Any

import pyarrow as pa
import pytest

from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.row_preserving.offset.sqlite_offset import SqliteOffset
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.row_preserving.offset.offset import (
    OffsetTestBase,
    make_feature_set,
)


class TestSqliteOffset(OffsetTestBase):
    @classmethod
    def supported_offset_types(cls) -> set[str]:
        return {"lag", "lead", "first_value", "last_value"}

    def setup_method(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        super().setup_method()

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteOffset

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        return SqliteRelation.from_arrow(self.conn, arrow_table)

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        return list(result.to_arrow_table().column(column_name).to_pylist())

    def get_row_count(self, result: Any) -> int:
        return len(result)

    def get_expected_type(self) -> type:
        return SqliteRelation


class TestSqliteUnsupportedTypes:
    """Verify that SQLite raises ValueError for unsupported offset types."""

    def setup_method(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        arrow_table = PyArrowDataOpsTestDataCreator.create()
        self.test_data = SqliteRelation.from_arrow(self.conn, arrow_table)

    def teardown_method(self) -> None:
        self.conn.close()

    def test_diff_raises_value_error(self) -> None:
        fs = make_feature_set("value_int__diff_1_offset", ["region"], "value_int")
        with pytest.raises(ValueError, match="Unsupported offset type for SQLite: diff_1"):
            SqliteOffset.calculate_feature(self.test_data, fs)

    def test_pct_change_raises_value_error(self) -> None:
        fs = make_feature_set("value_int__pct_change_1_offset", ["region"], "value_int")
        with pytest.raises(ValueError, match="Unsupported offset type for SQLite: pct_change_1"):
            SqliteOffset.calculate_feature(self.test_data, fs)
