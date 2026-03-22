"""Tests for PyArrowRank compute implementation."""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from mloda.community.feature_groups.data_operations.row_preserving.rank.pyarrow_rank import (
    PyArrowRank,
)
from mloda.testing.feature_groups.data_operations.row_preserving.rank.rank import (
    RankTestBase,
)


class TestPyArrowRank(RankTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowRank

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        return arrow_table

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        return list(result.column(column_name).to_pylist())

    def get_row_count(self, result: Any) -> int:
        return int(result.num_rows)

    def get_expected_type(self) -> Any:
        return pa.Table
