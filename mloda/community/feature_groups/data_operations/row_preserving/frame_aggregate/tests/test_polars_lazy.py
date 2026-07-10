"""Tests for Polars lazy frame aggregate implementation.

Uses the unified FrameAggregateTestBase.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("polars")

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.polars_lazy import PolarsLazyTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.frame_aggregate.frame_aggregate import (
    FrameAggregateTestBase,
    config_frame_options,
)

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.polars_lazy_frame_aggregate import (
    PolarsLazyFrameAggregate,
)


class TestPolarsLazyFrameAggregate(CapabilityHookTestMixin, PolarsLazyTestMixin, FrameAggregateTestBase):
    """Unified tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyFrameAggregate

    @classmethod
    def supports_null_order_in_time_window(cls) -> bool:
        # polars rolling_*_by(ts) panics on null timestamps in the order_by column.
        return False

    @classmethod
    def capability_supported(cls) -> tuple[tuple[str, Options], ...]:
        return (
            ("value__median_rolling_3", Options()),
            ("value__std_rolling_3", Options()),
            ("value__var_rolling_3", Options()),
            ("value__median_7_day_window", Options()),
            *(("value_frame", config_frame_options(t, "rolling")) for t in ("std", "var", "median")),
        )

    @classmethod
    def capability_unsupported(cls) -> tuple[tuple[str, Options], ...]:
        return (
            ("value__cumstd", Options()),
            ("value__cumvar", Options()),
            ("value__cummedian", Options()),
            ("value__expanding_std", Options()),
            ("value__expanding_var", Options()),
            ("value__expanding_median", Options()),
            *(
                ("value_frame", config_frame_options(t, ft))
                for t in ("std", "var", "median")
                for ft in ("cumulative", "expanding")
            ),
        )
