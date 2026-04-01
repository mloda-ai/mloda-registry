"""Tests for Polars lazy frame aggregate implementation.

Uses the unified FrameAggregateTestBase.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("polars")

from mloda.testing.feature_groups.data_operations.mixins.polars_lazy import PolarsLazyTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.frame_aggregate.frame_aggregate import (
    FrameAggregateTestBase,
)

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.polars_lazy_frame_aggregate import (
    PolarsLazyFrameAggregate,
)


class TestPolarsLazyFrameAggregate(PolarsLazyTestMixin, FrameAggregateTestBase):
    """Unified tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyFrameAggregate
