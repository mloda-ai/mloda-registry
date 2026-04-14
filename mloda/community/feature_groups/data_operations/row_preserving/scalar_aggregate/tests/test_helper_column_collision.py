"""Helper-column collision tests for scalar_aggregate implementations.

These tests verify that user-provided columns whose names happen to match
internal hardcoded helper-column names survive the computation unmodified.

Helper names currently used internally:
- Polars mask temp: ``__mloda_masked_src__``
"""

from __future__ import annotations

import pytest


class TestPolarsLazyScalarAggregateMaskCollision:
    """User column named ``__mloda_masked_src__`` must survive PolarsLazyScalarAggregate."""

    def test_user_column_named_masked_src_survives(self) -> None:
        pytest.importorskip("polars")
        import polars as pl

        from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.polars_lazy_scalar_aggregate import (
            PolarsLazyScalarAggregate,
        )

        data = pl.LazyFrame(
            {
                "value": [10.0, 20.0, 30.0, 40.0, 50.0],
                "flag": ["A", "A", "B", "A", "A"],
                "__mloda_masked_src__": ["u0", "u1", "u2", "u3", "u4"],
            }
        )

        result = PolarsLazyScalarAggregate._compute_aggregation(
            data=data,
            feature_name="sum_value",
            source_col="value",
            agg_type="sum",
            mask_spec=[("flag", "equal", "A")],
        ).collect()

        # Mask keeps rows where flag == "A": 10+20+40+50 = 120
        assert "__mloda_masked_src__" in result.columns
        assert result["__mloda_masked_src__"].to_list() == ["u0", "u1", "u2", "u3", "u4"]
        assert "sum_value" in result.columns
        assert result["sum_value"].to_list() == [120.0, 120.0, 120.0, 120.0, 120.0]
