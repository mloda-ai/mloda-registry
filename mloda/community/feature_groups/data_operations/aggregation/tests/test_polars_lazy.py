"""Tests for PolarsLazyColumnAggregation compute implementation."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

pytest.importorskip("polars")

import polars as pl

from mloda.community.feature_groups.data_operations.aggregation.polars_lazy_aggregation import (
    PolarsLazyColumnAggregation,
)
from mloda.testing.feature_groups.data_operations.aggregation import (
    AggregationTestBase,
)


class TestPolarsLazyColumnAggregation(AggregationTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyColumnAggregation

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        return pl.from_arrow(arrow_table).lazy()

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        collected = result.collect()
        return list(collected[column_name].to_list())

    def get_row_count(self, result: Any) -> int:
        return int(result.collect().height)

    def get_expected_type(self) -> Any:
        return pl.LazyFrame
