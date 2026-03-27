"""Tests for Polars lazy frame aggregate implementation.

Uses the unified FrameAggregateTestBase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

pl = pytest.importorskip("polars")

if TYPE_CHECKING:
    import polars

import pyarrow as pa

from mloda.testing.feature_groups.data_operations.row_preserving.frame_aggregate import (
    FrameAggregateTestBase,
)

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.polars_lazy_frame_aggregate import (
    PolarsLazyFrameAggregate,
)


class TestPolarsLazyFrameAggregate(FrameAggregateTestBase):
    """Unified tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyFrameAggregate

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        return pl.from_arrow(arrow_table).lazy()

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        df = result.collect()
        return list(df[column_name].to_list())

    def get_row_count(self, result: Any) -> int:
        return len(result.collect())

    def get_expected_type(self) -> Any:
        return pl.LazyFrame
