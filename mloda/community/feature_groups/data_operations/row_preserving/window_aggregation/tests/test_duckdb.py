"""Tests for DuckdbWindowAggregation compute implementation."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

duckdb = pytest.importorskip("duckdb")

from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.duckdb_window_aggregation import (
    DuckdbWindowAggregation,
)
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.pyarrow_window_aggregation import (
    PyArrowWindowAggregation,
)
from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation import (
    WindowAggregationTestBase,
)


class TestDuckdbWindowAggregation(WindowAggregationTestBase):
    """Standard tests inherited from the base class."""

    def setup_method(self) -> None:
        self.conn = duckdb.connect()
        super().setup_method()

    def teardown_method(self) -> None:
        self.conn.close()

    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbWindowAggregation

    @classmethod
    def pyarrow_implementation_class(cls) -> Any:
        return PyArrowWindowAggregation

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        return DuckdbRelation.from_arrow(self.conn, arrow_table)

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        return list(result.to_arrow_table().column(column_name).to_pylist())

    def get_row_count(self, result: Any) -> int:
        return int(result.to_arrow_table().num_rows)

    def get_expected_type(self) -> Any:
        return DuckdbRelation
