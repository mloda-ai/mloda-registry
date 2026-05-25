"""Tests for Pandas frame aggregate implementation.

Uses the unified FrameAggregateTestBase.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pyarrow as pa
import pytest

pytest.importorskip("pandas")

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.helpers import (
    extract_column as _extract_column,
    make_feature_set,
)
from mloda.testing.feature_groups.data_operations.mixins.pandas import PandasTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.frame_aggregate.frame_aggregate import (
    FrameAggregateTestBase,
)
from mloda.user import Feature

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

    # -- Regression tests for PR #202 review bugs -----------------------------

    def test_pandas_calendar_time_handles_duplicate_index(self) -> None:
        """Duplicate index labels (e.g. from pd.concat) must not collapse positional writes."""
        import pandas as pd

        table = pa.table(
            {
                "region": ["A", "A", "A", "A"],
                "ts": [
                    datetime(2023, 1, 1, tzinfo=timezone.utc),
                    datetime(2023, 2, 1, tzinfo=timezone.utc),
                    datetime(2023, 3, 1, tzinfo=timezone.utc),
                    datetime(2023, 4, 1, tzinfo=timezone.utc),
                ],
                "value": [10, 20, 30, 40],
            }
        )
        df = table.to_pandas()
        # Force duplicate index labels (simulates pd.concat with overlapping integer indices).
        df.index = pd.Index([0, 0, 1, 1])
        feature_name = "value__sum_1_month_window"
        feature = Feature(
            feature_name,
            options=Options(context={"partition_by": ["region"], "order_by": "ts"}),
        )
        fs = FeatureSet()
        fs.add(feature)
        result = self.implementation_class().calculate_feature(df, fs)
        ref = self.reference_implementation_class().calculate_feature(table, fs)
        result_col = [None if pd.isna(v) else v for v in result[feature_name].tolist()]
        ref_col = _extract_column(ref, feature_name)
        assert result_col == ref_col, f"got {result_col}, expected {ref_col}"

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

    def test_pandas_time_window_source_equals_order_with_mask(self) -> None:
        """source_col == order_by + mask must not corrupt order_by.

        Mask writes NaN into source_col; if source_col is also order_by, .rolling(on=ts)
        rejects the resulting non-monotonic series.
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
        # Should not raise; should match reference.
        result = self.implementation_class().calculate_feature(data, fs)
        ref = self.reference_implementation_class().calculate_feature(table, fs)
        result_col = self.extract_column(result, feature_name)
        ref_col = _extract_column(ref, feature_name)
        assert result_col == ref_col, f"got {result_col}, expected {ref_col}"
