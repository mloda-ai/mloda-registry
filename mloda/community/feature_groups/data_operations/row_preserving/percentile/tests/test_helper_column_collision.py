"""Helper-column collision tests for percentile implementations.

These tests verify that user-provided columns whose names happen to match
internal hardcoded helper-column names survive the computation unmodified.

ReferencePercentile uses a ``__mloda_pctl_idx__`` helper column.

Framework-agnostic assertions live in
``mloda.testing.feature_groups.data_operations.collision``.
"""

from __future__ import annotations

import pytest

from mloda.testing.feature_groups.data_operations.collision import assert_collision_preserved

_USER_VALUES = ["u0", "u1", "u2", "u3", "u4"]


class TestReferencePercentileCollision:
    """User column named ``__mloda_pctl_idx__`` must survive ReferencePercentile."""

    def test_user_column_named_pctl_idx_survives(self) -> None:
        pa = pytest.importorskip("pyarrow")

        from mloda.testing.feature_groups.data_operations.row_preserving.percentile.reference import (
            ReferencePercentile,
        )

        table = pa.table(
            {
                "region": ["A", "A", "A", "B", "B"],
                "value": [10.0, 20.0, 30.0, 40.0, 50.0],
                "__mloda_pctl_idx__": _USER_VALUES,
            }
        )

        result = ReferencePercentile._compute_percentile(
            table=table,
            feature_name="p50_value",
            source_col="value",
            partition_by=["region"],
            percentile=0.5,
        )

        assert_collision_preserved(
            result, "__mloda_pctl_idx__", _USER_VALUES, "p50_value", [20.0, 20.0, 20.0, 45.0, 45.0]
        )


class TestPolarsLazyPercentileMaskCollision:
    """User column named ``__mloda_masked_src__`` must survive PolarsLazyPercentile."""

    def test_user_column_named_masked_src_survives(self) -> None:
        pytest.importorskip("polars")
        import polars as pl

        from mloda.community.feature_groups.data_operations.row_preserving.percentile.polars_lazy_percentile import (
            PolarsLazyPercentile,
        )

        data = pl.LazyFrame(
            {
                "region": ["A", "A", "A", "B", "B"],
                "value": [10.0, 20.0, 30.0, 40.0, 50.0],
                "flag": ["A", "A", "B", "A", "A"],
                "__mloda_masked_src__": _USER_VALUES,
            }
        )

        result = PolarsLazyPercentile._compute_percentile(
            data=data,
            feature_name="p50_value",
            source_col="value",
            partition_by=["region"],
            percentile=0.5,
            mask_spec=[("flag", "equal", "A")],
        )

        # Masked values per region:
        #   region A: [10, 20, null] -> median of non-null = 15
        #   region B: [40, 50]       -> median = 45
        assert_collision_preserved(
            result, "__mloda_masked_src__", _USER_VALUES, "p50_value", [15.0, 15.0, 15.0, 45.0, 45.0]
        )
