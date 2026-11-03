"""Tests for SqliteOffset compute implementation."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from mloda.community.feature_groups.data_operations.row_preserving.offset.sqlite_offset import SqliteOffset
from mloda.testing.feature_groups.data_operations.helpers import calculate_with, make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.sqlite import SqliteTestMixin, SqliteViewTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.offset.offset import (
    OffsetTestBase,
)


class TestSqliteOffset(SqliteTestMixin, OffsetTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteOffset

    @pytest.mark.parametrize("op", ["first_value", "last_value"])
    def test_all_null_int_source_keeps_int64_type(self, op: str) -> None:
        # Matches DuckDB, pinned.
        table = pa.table({"region": ["A"] * 3, "score": pa.array([None] * 3, type=pa.int64())})
        data = self.create_test_data(table)
        fs = make_feature_set(f"score__{op}_offset", ["region"], "score")
        result = calculate_with(self.implementation_class(), data, fs)

        assert result.to_arrow_table().schema.field(f"score__{op}_offset").type == pa.int64()


class TestSqliteOffsetViewInput(SqliteViewTestMixin, TestSqliteOffset):
    """Runs the Offset suite on a view input."""
