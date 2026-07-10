"""Tests for Pandas frame aggregate implementation.

Uses the unified FrameAggregateTestBase.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pandas")

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.pandas import PandasTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.frame_aggregate.frame_aggregate import (
    FrameAggregateTestBase,
    time_frame_options,
)

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.pandas_frame_aggregate import (
    PandasFrameAggregate,
)


class TestPandasFrameAggregate(CapabilityHookTestMixin, PandasTestMixin, FrameAggregateTestBase):
    """Unified tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PandasFrameAggregate

    @classmethod
    def capability_supported(cls) -> tuple[tuple[str, Options], ...]:
        return (
            ("value_time_frame", time_frame_options("day")),
            ("value__sum_rolling_3", Options()),
            ("value__median_rolling_3", Options()),
        )

    @classmethod
    def capability_unsupported(cls) -> tuple[tuple[str, Options], ...]:
        return (("value_time_frame", time_frame_options("month")),)

    @classmethod
    def supported_time_units(cls) -> set[str]:
        # Pandas .rolling(window="...", on=ts) accepts only fixed-frequency units.
        # Month/year are calendar-anchored; rather than fall back to a Python loop
        # (which would defeat the point of running inside pandas), they are rejected
        # at match time. See known-divergences.md.
        return {"second", "minute", "hour", "day", "week"}

    @classmethod
    def supports_null_order_in_time_window(cls) -> bool:
        # pandas groupby().rolling(on=ts) raises "ts values must not have NaT"
        # when the order_by column contains null timestamps. The implementation
        # surfaces this as an explicit ValueError before calling rolling().
        return False

    # -- Regression tests for PR #202 review bugs -----------------------------

    def test_pandas_rejects_month_time_window_at_match_time(self) -> None:
        """Pandas rejects month/year time windows because they would require a Python loop."""
        options = Options(context={"partition_by": ["region"], "order_by": "ts"})
        assert not self.implementation_class().match_feature_group_criteria("value__sum_1_month_window", options)
        assert not self.implementation_class().match_feature_group_criteria("value__sum_1_year_window", options)
        # day/week still match
        assert self.implementation_class().match_feature_group_criteria("value__sum_3_day_window", options)
        assert self.implementation_class().match_feature_group_criteria("value__sum_1_week_window", options)

    def test_pandas_dep_pinned_to_22_for_lowercase_freq_codes(self) -> None:
        """_FIXED_FREQ_CODES uses lowercase 's','min','h' which require pandas>=2.2."""
        try:
            import tomllib  # py311+
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef,unused-ignore]
        from pathlib import Path

        repo_root = Path(__file__).resolve()
        # Walk up to find config/packages.toml.
        for _ in range(10):
            repo_root = repo_root.parent
            if (repo_root / "config" / "packages.toml").exists():
                break
        cfg = tomllib.loads((repo_root / "config" / "packages.toml").read_text())
        pandas_deps = cfg["packages"]["mloda-community-frame-aggregate"]["optional_dependencies"]["pandas"]
        assert any(">=2.2" in d or ">=2.3" in d for d in pandas_deps), (
            f"frame-aggregate pandas dep should pin >=2.2 for lowercase freq codes; got {pandas_deps}"
        )
