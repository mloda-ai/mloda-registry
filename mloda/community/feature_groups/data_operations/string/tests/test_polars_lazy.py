"""Tests for PolarsLazyStringOps compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("polars")

import polars as pl
import pyarrow as pa

from mloda.community.feature_groups.data_operations.string.polars_lazy_string import (
    PolarsLazyStringOps,
)
from mloda.testing.feature_groups.data_operations.string import (
    StringTestBase,
)


class TestPolarsLazyStringOps(StringTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyStringOps

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        return pl.from_arrow(arrow_table).lazy()

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        collected = result.collect()
        return list(collected[column_name].to_list())

    def get_row_count(self, result: Any) -> int:
        return int(result.collect().height)

    def get_expected_type(self) -> Any:
        return pl.LazyFrame
