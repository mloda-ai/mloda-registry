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

    @pytest.mark.parametrize("source_type", [pa.int64(), pa.float64()])
    @pytest.mark.parametrize("op", ["first_value", "last_value"])
    def test_all_null_source_keeps_source_type(self, op: str, source_type: pa.DataType) -> None:
        # Keeps the source type, as DuckDB does; the reference gives a null-typed column.
        table = pa.table({"region": ["A"] * 3, "score": pa.array([None] * 3, type=source_type)})
        data = self.create_test_data(table)
        fs = make_feature_set(f"score__{op}_offset", ["region"], "score")
        result = calculate_with(self.implementation_class(), data, fs)

        assert result.to_arrow_table().schema.field(f"score__{op}_offset").type == source_type

    @pytest.mark.parametrize(
        "op, source_type, expected",
        [
            pytest.param(
                op,
                source_type,
                pa.float64() if op == "pct_change_1" else source_type,
                id=f"{op}-{source_type}",
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="https://github.com/mloda-ai/mloda/issues/2561: SQLite type probe types offset windows as string on an all-null source",
                ),
            )
            for op in ["lag_1", "lead_1", "diff_1", "pct_change_1"]
            for source_type in [pa.int64(), pa.float64()]
        ],
    )
    def test_all_null_source_keeps_type_for_row_offsets(
        self, op: str, source_type: pa.DataType, expected: pa.DataType
    ) -> None:
        table = pa.table({"region": ["A"] * 3, "score": pa.array([None] * 3, type=source_type)})
        data = self.create_test_data(table)
        fs = make_feature_set(f"score__{op}_offset", ["region"], "score")
        result = calculate_with(self.implementation_class(), data, fs)

        assert result.to_arrow_table().schema.field(f"score__{op}_offset").type == expected


class TestSqliteOffsetViewInput(SqliteViewTestMixin, TestSqliteOffset):
    """Runs the Offset suite on a view input."""
