"""Helper-column collision tests for aggregation implementations.

These tests verify that user-provided columns whose names happen to match
internal hardcoded helper-column names survive the computation. For
group_by aggregation the user column cannot be preserved positionally
(group_by collapses rows), but the implementation must not raise and the
masked aggregate must still be computed correctly, and the temp helper
column must not leak into the grouped result.

Helper names currently used internally:
- Polars mask temp: ``__mloda_masked_src__``

Framework-agnostic assertions live in
``mloda.testing.feature_groups.data_operations.collision``.
"""

from __future__ import annotations

import pytest

from mloda.testing.feature_groups.data_operations.collision import assert_column_absent

_USER_VALUES = ["u0", "u1", "u2", "u3", "u4"]


class TestPolarsLazyAggregationMaskCollision:
    """User column named ``__mloda_masked_src__`` must not break PolarsLazyAggregation.

    When a user column happens to share the name of the internal mask temp
    column, the implementation must still compute the correct masked
    aggregate and must not raise.
    """

    def test_user_column_named_masked_src_does_not_break_aggregate(self) -> None:
        pytest.importorskip("polars")
        import polars as pl

        from mloda.community.feature_groups.data_operations.aggregation.polars_lazy_aggregation import (
            PolarsLazyAggregation,
        )

        data = pl.LazyFrame(
            {
                "region": ["A", "A", "A", "B", "B"],
                "value": [10.0, 20.0, 30.0, 40.0, 50.0],
                "flag": ["A", "A", "B", "A", "A"],
                "__mloda_masked_src__": _USER_VALUES,
            }
        )

        result = PolarsLazyAggregation._compute_group(
            data=data,
            feature_name="sum_value",
            source_col="value",
            partition_by=["region"],
            agg_type="sum",
            mask_spec=[("flag", "equal", "A")],
        ).collect()

        # Mask keeps rows where flag == "A": region A -> 10+20 = 30, region B -> 40+50 = 90
        rows = {r["region"]: r["sum_value"] for r in result.to_dicts()}
        assert rows["A"] == 30.0
        assert rows["B"] == 90.0
        # Temp mask column must not leak into the grouped result
        assert_column_absent(result, "__mloda_masked_src__")
        assert "sum_value" in result.columns
