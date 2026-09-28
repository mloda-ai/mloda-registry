"""Tests for DuckDB frame aggregate implementation.

Uses the unified FrameAggregateTestBase.
"""

from __future__ import annotations

from typing import Any

import pytest

duckdb = pytest.importorskip("duckdb")

from mloda.user import Options

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.duckdb_frame_aggregate import (
    DuckdbFrameAggregate,
)
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.duckdb import DuckdbTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.frame_aggregate.frame_aggregate import (
    FrameAggregateTestBase,
    time_frame_options,
)


class TestDuckdbFrameAggregate(CapabilityHookTestMixin, DuckdbTestMixin, FrameAggregateTestBase):
    """Unified tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbFrameAggregate

    @classmethod
    def capability_supported(cls) -> tuple[tuple[str, Options], ...]:
        return (
            ("value_time_frame", time_frame_options("month")),
            ("value__median_rolling_3", Options()),
        )

    def nan_policy_skip_if_unsupported(self, case: str, agg_type: str, feature_name: str) -> None:
        """Pins the DuckDB crash so the skip fails once DuckDB propagates NaN instead of raising."""
        super().nan_policy_skip_if_unsupported(case, agg_type, feature_name)
        if agg_type in {"std", "var"}:
            table = self.nan_policy_table()
            fs = self.nan_policy_feature_set(feature_name)
            with pytest.raises(duckdb.OutOfRangeException):
                self.extract_column(
                    self.implementation_class().calculate_feature(self.create_test_data(table), fs),
                    feature_name,
                )
            pytest.skip("DuckDB STDDEV_POP/VAR_POP raise OutOfRangeException on NaN input")
