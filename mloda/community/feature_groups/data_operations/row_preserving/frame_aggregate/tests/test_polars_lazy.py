"""Tests for Polars lazy frame aggregate implementation.

Uses the unified FrameAggregateTestBase.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("polars")

from mloda.testing.feature_groups.data_operations.mixins.polars_lazy import PolarsLazyTestMixin
from mloda.testing.feature_groups.data_operations.mixins.reserved_columns import ReservedColumnsTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.frame_aggregate.frame_aggregate import (
    FrameAggregateTestBase,
)

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.polars_lazy_frame_aggregate import (
    PolarsLazyFrameAggregate,
)


class TestPolarsLazyFrameAggregate(ReservedColumnsTestMixin, PolarsLazyTestMixin, FrameAggregateTestBase):
    """Unified tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyFrameAggregate

    @classmethod
    def reserved_columns_feature_name(cls) -> str:
        return "value_int__sum_rolling_3"

    @classmethod
    def reserved_columns_partition_by(cls) -> list[str] | None:
        return ["region"]

    @classmethod
    def reserved_columns_order_by(cls) -> str | None:
        return "timestamp"
