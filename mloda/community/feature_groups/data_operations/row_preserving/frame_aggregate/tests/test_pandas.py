"""Tests for Pandas frame aggregate implementation.

Uses the unified FrameAggregateTestBase.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pyarrow as pa
import pytest

pytest.importorskip("pandas")

from mloda.user import Options

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.pandas_frame_aggregate import (
    PandasFrameAggregate,
)
from mloda.testing.feature_groups.data_operations.helpers import calculate_with, feature_group_of, make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.pandas import (
    PandasNullableIntTestMixin,
    PandasTestMixin,
)
from mloda.testing.feature_groups.data_operations.row_preserving.frame_aggregate.frame_aggregate import (
    FrameAggregateTestBase,
    time_frame_options,
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

    @classmethod
    def supports_datetime_min_max_source(cls) -> bool:
        # pandas window ops only aggregate numeric columns, so a datetime min/max source is rejected.
        return False

    # -- Regression tests for PR #202 review bugs -----------------------------

    def test_pandas_rejects_month_time_window_at_match_time(self) -> None:
        """Pandas rejects month/year time windows because they would require a Python loop."""
        options = Options(context={"partition_by": ["region"], "order_by": "ts"})
        group = feature_group_of(self.implementation_class())
        framework = self.compute_framework_class()
        assert not group.supports_compute_framework("value__sum_1_month_window", options, framework)
        assert not group.supports_compute_framework("value__sum_1_year_window", options, framework)
        # day/week still match
        assert group.supports_compute_framework("value__sum_3_day_window", options, framework)
        assert group.supports_compute_framework("value__sum_1_week_window", options, framework)

    @pytest.mark.parametrize(
        "agg_type, source_kind",
        [
            pytest.param("min", "datetime", id="min"),
            pytest.param("max", "datetime", id="max"),
            pytest.param("min", "timedelta", id="timedelta_min"),
            pytest.param("max", "timedelta", id="timedelta_max"),
            pytest.param("sum", "timedelta", id="timedelta_sum"),
        ],
    )
    def test_pandas_rejects_datetime_min_max_in_rolling_frame(self, agg_type: str, source_kind: str) -> None:
        """A non-time rolling frame over a datetime or timedelta source is rejected with a ValueError too."""
        timestamps = [datetime(2023, 1, d, tzinfo=timezone.utc) for d in (1, 3, 5)]
        column = "ts" if source_kind == "datetime" else "dur"
        data: dict[str, Any] = {"region": ["A", "A", "A"], "ts": timestamps}
        if source_kind == "timedelta":
            data["dur"] = [timedelta(days=d) for d in (1, 3, 5)]
        table = pa.table(data)
        fs = make_feature_set(f"{column}__{agg_type}_rolling_3", partition_by=["region"], order_by="ts")
        with pytest.raises(ValueError, match=rf"{source_kind}.*{column}|{column}.*{source_kind}"):
            calculate_with(self.implementation_class(), self.create_test_data(table), fs)

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


class TestPandasFrameAggregateNullableInt(PandasNullableIntTestMixin, TestPandasFrameAggregate):
    """Runs the frame_aggregate suite on nullable pandas int columns."""
