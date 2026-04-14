"""Helper-column collision tests for percentile implementations.

These tests verify that user-provided columns whose names happen to match
internal hardcoded helper-column names survive the computation unmodified.

ReferencePercentile uses a ``__mloda_pctl_idx__`` helper column.
"""

from __future__ import annotations

import pytest


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
                "__mloda_pctl_idx__": ["u0", "u1", "u2", "u3", "u4"],
            }
        )

        result = ReferencePercentile._compute_percentile(
            table=table,
            feature_name="p50_value",
            source_col="value",
            partition_by=["region"],
            percentile=0.5,
        )

        assert "__mloda_pctl_idx__" in result.column_names
        assert result.column("__mloda_pctl_idx__").to_pylist() == ["u0", "u1", "u2", "u3", "u4"]
        assert "p50_value" in result.column_names
        assert result.column("p50_value").to_pylist() == [20.0, 20.0, 20.0, 45.0, 45.0]


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
                "__mloda_masked_src__": ["u0", "u1", "u2", "u3", "u4"],
            }
        )

        result = PolarsLazyPercentile._compute_percentile(
            data=data,
            feature_name="p50_value",
            source_col="value",
            partition_by=["region"],
            percentile=0.5,
            mask_spec=[("flag", "equal", "A")],
        ).collect()

        # Masked values per region:
        #   region A: [10, 20, null] -> median of non-null = 15
        #   region B: [40, 50]       -> median = 45
        assert "__mloda_masked_src__" in result.columns
        assert result["__mloda_masked_src__"].to_list() == ["u0", "u1", "u2", "u3", "u4"]
        assert "p50_value" in result.columns
        assert result["p50_value"].to_list() == [15.0, 15.0, 15.0, 45.0, 45.0]
