"""Tests for DuckdbWindowAggregation compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

duckdb = pytest.importorskip("duckdb")

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.duckdb_window_aggregation import (
    DuckdbWindowAggregation,
)
from mloda.testing.feature_groups.data_operations.mixins.duckdb import DuckdbTestMixin
from mloda.testing.feature_groups.data_operations.mixins.reserved_columns import ReservedColumnsTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation.window_aggregation import (
    WindowAggregationTestBase,
)


class TestDuckdbWindowAggregation(ReservedColumnsTestMixin, DuckdbTestMixin, WindowAggregationTestBase):
    """Standard tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbWindowAggregation

    @classmethod
    def reserved_columns_feature_name(cls) -> str:
        return "value_int__sum_window"

    @classmethod
    def reserved_columns_partition_by(cls) -> list[str] | None:
        return ["region"]

    @classmethod
    def reserved_columns_order_by(cls) -> str | None:
        return None
