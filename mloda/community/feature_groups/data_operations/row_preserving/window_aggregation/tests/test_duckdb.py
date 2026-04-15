"""Tests for DuckdbWindowAggregation compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

duckdb = pytest.importorskip("duckdb")

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.duckdb_window_aggregation import (
    DuckdbWindowAggregation,
)
from mloda.testing.feature_groups.data_operations.mixins.duckdb import DuckdbTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation.window_aggregation import (
    WindowAggregationTestBase,
)


class TestDuckdbWindowAggregation(DuckdbTestMixin, WindowAggregationTestBase):
    """Standard tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbWindowAggregation

    def test_collision_rn(self) -> None:
        """User column named __mloda_rn__ must survive DuckdbWindowAggregation first/last.

        The __mloda_rn__ helper is only materialized by the first/last code
        paths, so this test uses the ``first`` aggregation rather than the
        default ``sum`` baseline.
        """
        self._run_collision_case(
            "__mloda_rn__",
            feature_name="value_int__first_window",
            partition_by=["region"],
            order_by="value_int",
        )
