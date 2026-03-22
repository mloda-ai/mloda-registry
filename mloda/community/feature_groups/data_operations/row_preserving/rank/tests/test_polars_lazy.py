"""Tests for PolarsLazyRank compute implementation."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

pytest.importorskip("polars")

import polars as pl

from mloda.community.feature_groups.data_operations.row_preserving.rank.polars_lazy_rank import (
    PolarsLazyRank,
)
from mloda.testing.feature_groups.data_operations.row_preserving.rank.rank import (
    RankTestBase,
)


class TestPolarsLazyRank(RankTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyRank

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        return pl.from_arrow(arrow_table).lazy()

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        collected = result.collect()
        return list(collected[column_name].to_list())

    def get_row_count(self, result: Any) -> int:
        return int(result.collect().height)

    def get_expected_type(self) -> Any:
        return pl.LazyFrame
