"""Tests for Pandas frame aggregate implementation.

Uses the unified FrameAggregateTestBase.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pandas")

from mloda.testing.feature_groups.data_operations.mixins.pandas import PandasTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.frame_aggregate.frame_aggregate import (
    FrameAggregateTestBase,
)

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.pandas_frame_aggregate import (
    PandasFrameAggregate,
)


class TestPandasFrameAggregate(PandasTestMixin, FrameAggregateTestBase):
    """Unified tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PandasFrameAggregate

    @classmethod
    def supports_null_order_in_time_window(cls) -> bool:
        # pandas groupby().rolling(on=ts) raises "ts values must not have NaT"
        # when the order_by column contains null timestamps.
        return False
