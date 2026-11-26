"""Tests for SqliteOffset compute implementation."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest
from mloda.user import DataType

from mloda.community.feature_groups.data_operations.row_preserving.offset.sqlite_offset import SqliteOffset
from mloda.testing.feature_groups.data_operations.helpers import calculate_with, make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.frameworks import SqliteTestMixin, SqliteViewTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.offset.offset import (
    OffsetTestBase,
)
from mloda.testing.typed_data import TypedTestData


class TestSqliteOffset(SqliteTestMixin, OffsetTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteOffset

    @classmethod
    def holds_uint64_above_int64(cls) -> bool:
        return False

    @pytest.mark.parametrize("source_type", [pa.int64(), pa.float64()])
    @pytest.mark.parametrize("op", ["first_value", "last_value"])
    def test_all_null_source_keeps_source_type(self, op: str, source_type: pa.DataType) -> None:
        # Keeps the source type, as DuckDB does; the reference gives a null-typed column.
        table = TypedTestData({"region": (DataType.STRING, ["A"] * 3), "score": (DataType.INT64, [None] * 3)})
        data = self.create_test_data(table, {"score": source_type})  # parametrized physical type
        fs = make_feature_set(f"score__{op}_offset", ["region"], "score")
        result = calculate_with(self.implementation_class(), data, fs)

        assert result.to_arrow_table().schema.field(f"score__{op}_offset").type == source_type

    @pytest.mark.parametrize("source_type", [pa.int64(), pa.float64()])
    @pytest.mark.parametrize("op", ["lag_1", "lead_1", "diff_1", "pct_change_1"])
    def test_all_null_source_keeps_type_for_row_offsets(self, op: str, source_type: pa.DataType) -> None:
        expected = pa.float64() if op == "pct_change_1" else source_type
        table = TypedTestData({"region": (DataType.STRING, ["A"] * 3), "score": (DataType.INT64, [None] * 3)})
        data = self.create_test_data(table, {"score": source_type})  # parametrized physical type
        fs = make_feature_set(f"score__{op}_offset", ["region"], "score")
        result = calculate_with(self.implementation_class(), data, fs)

        assert result.to_arrow_table().schema.field(f"score__{op}_offset").type == expected


class TestSqliteOffsetViewInput(SqliteViewTestMixin, TestSqliteOffset):
    """Runs the Offset suite on a view input."""
