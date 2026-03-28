"""Tests for PolarsLazyDateTimeExtraction compute implementation."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

pytest.importorskip("polars")

import polars as pl

from mloda.community.feature_groups.data_operations.row_preserving.datetime.polars_lazy_datetime import (
    PolarsLazyDateTimeExtraction,
)
from mloda.testing.feature_groups.data_operations.row_preserving.datetime import (
    DateTimeTestBase,
)


class TestPolarsLazyDateTimeExtraction(DateTimeTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyDateTimeExtraction

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        return pl.from_arrow(arrow_table).lazy()

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        collected = result.collect()
        return [None if v is None else v for v in collected[column_name].to_list()]

    def get_row_count(self, result: Any) -> int:
        return int(result.collect().height)

    def get_expected_type(self) -> Any:
        return pl.LazyFrame
