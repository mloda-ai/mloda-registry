"""Tests for PyArrowGroupAggregation compute implementation."""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from mloda.community.feature_groups.data_operations.group_aggregation.pyarrow_group_aggregation import (
    PyArrowGroupAggregation,
)
from mloda.testing.feature_groups.data_operations.group_aggregation.group_aggregation import (
    GroupAggregationTestBase,
)


class TestPyArrowGroupAggregation(GroupAggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowGroupAggregation

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        return arrow_table

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        return list(result.column(column_name).to_pylist())

    def get_row_count(self, result: Any) -> int:
        return int(result.num_rows)

    def get_expected_type(self) -> Any:
        return pa.Table
