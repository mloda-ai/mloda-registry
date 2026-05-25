"""Tests for Pandas frame aggregate implementation.

Uses the unified FrameAggregateTestBase.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pyarrow as pa
import pytest

pytest.importorskip("pandas")

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
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
        import tomllib
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

    def test_pandas_time_window_source_equals_order_with_mask_rejected(self) -> None:
        """source_col == order_by + mask + time frame is rejected at runtime.

        The reference semantic treats masked rows as having null ``order_by`` (because
        mask writes null into source_col, which is also order_by). Pandas' native
        ``rolling(on=ts)`` cannot simulate this without a Python loop, so the
        implementation refuses the combo with a clear ValueError.
        """
        table = pa.table(
            {
                "region": ["A"] * 5,
                "ts": [datetime(2023, 1, d, tzinfo=timezone.utc) for d in (1, 3, 5, 7, 10)],
                "category": ["X", "Y", "X", "X", "Y"],
            }
        )
        data = self.create_test_data(table)
        # source == order == "ts"; mask on a third column.
        feature_name = "ts__count_3_day_window"
        fs = make_feature_set(
            feature_name,
            partition_by=["region"],
            order_by="ts",
            mask=("category", "equal", "X"),
        )
        with pytest.raises(ValueError, match="source_col == order_by"):
            self.implementation_class().calculate_feature(data, fs)
