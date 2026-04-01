"""PyArrow framework test mixin."""

from __future__ import annotations

from typing import Any

import pyarrow as pa


class PyArrowTestMixin:
    """Mixin implementing adapter methods for PyArrow."""

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        return arrow_table

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        return list(result.column(column_name).to_pylist())

    def get_row_count(self, result: Any) -> int:
        return int(result.num_rows)

    def get_expected_type(self) -> Any:
        return pa.Table
