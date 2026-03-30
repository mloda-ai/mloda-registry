"""Tests for DuckDB frame aggregate implementation.

Uses the unified FrameAggregateTestBase.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

duckdb = pytest.importorskip("duckdb")

from mloda.testing.feature_groups.data_operations.row_preserving.frame_aggregate import (
    FrameAggregateTestBase,
)
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.duckdb_frame_aggregate import (
    DuckDBFrameAggregate,
)


class TestDuckDBFrameAggregate(FrameAggregateTestBase):
    """Unified tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return DuckDBFrameAggregate

    def setup_method(self) -> None:
        self.conn = duckdb.connect(":memory:")
        super().setup_method()

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        rel = self.conn.from_arrow(arrow_table)
        return DuckdbRelation(self.conn, rel)

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        arrow_table = result.to_arrow_table()
        return list(arrow_table.column(column_name).to_pylist())

    def get_row_count(self, result: Any) -> int:
        return int(result.to_arrow_table().num_rows)

    def get_expected_type(self) -> Any:
        return DuckdbRelation
