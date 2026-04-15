"""Helper-column collision tests for frame_aggregate implementations.

These tests verify that user-provided columns whose names happen to match
internal hardcoded helper-column names (e.g. ``__mloda_rn__``) survive the
computation unmodified.

Framework-agnostic assertions live in
``mloda.testing.feature_groups.data_operations.collision``.
"""

from __future__ import annotations

import pytest

from mloda.testing.feature_groups.data_operations.collision import assert_collision_preserved

_USER_VALUES = ["u0", "u1", "u2", "u3", "u4"]


class TestPandasFrameAggregateCollision:
    """User column named ``__mloda_rn__`` must survive PandasFrameAggregate."""

    def test_user_column_named_rn_survives(self) -> None:
        pytest.importorskip("pandas")
        import pandas as pd

        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.pandas_frame_aggregate import (
            PandasFrameAggregate,
        )

        data = pd.DataFrame(
            {
                "region": ["A", "A", "A", "B", "B"],
                "ts": [1, 2, 3, 1, 2],
                "value": [10.0, 20.0, 30.0, 40.0, 50.0],
                "__mloda_rn__": _USER_VALUES,
            }
        )

        result = PandasFrameAggregate._compute_frame(
            data=data,
            feature_name="sum_value",
            source_col="value",
            partition_by=["region"],
            order_by="ts",
            agg_type="sum",
            frame_type="cumulative",
        )

        assert_collision_preserved(result, "__mloda_rn__", _USER_VALUES, "sum_value", [10.0, 30.0, 60.0, 40.0, 90.0])


class TestPolarsLazyFrameAggregateCollision:
    """User column named ``__mloda_rn__`` must survive PolarsLazyFrameAggregate."""

    def test_user_column_named_rn_survives(self) -> None:
        pytest.importorskip("polars")
        import polars as pl

        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.polars_lazy_frame_aggregate import (
            PolarsLazyFrameAggregate,
        )

        data = pl.LazyFrame(
            {
                "region": ["A", "A", "A", "B", "B"],
                "ts": [1, 2, 3, 1, 2],
                "value": [10.0, 20.0, 30.0, 40.0, 50.0],
                "__mloda_rn__": _USER_VALUES,
            }
        )

        result = PolarsLazyFrameAggregate._compute_frame(
            data=data,
            feature_name="sum_value",
            source_col="value",
            partition_by=["region"],
            order_by="ts",
            agg_type="sum",
            frame_type="cumulative",
        )

        assert_collision_preserved(result, "__mloda_rn__", _USER_VALUES, "sum_value", [10.0, 30.0, 60.0, 40.0, 90.0])


class TestPolarsLazyFrameAggregateMaskCollision:
    """User column named ``__mloda_masked_src__`` must survive PolarsLazyFrameAggregate."""

    def test_user_column_named_masked_src_survives(self) -> None:
        pytest.importorskip("polars")
        import polars as pl

        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.polars_lazy_frame_aggregate import (
            PolarsLazyFrameAggregate,
        )

        data = pl.LazyFrame(
            {
                "region": ["A", "A", "A", "B", "B"],
                "ts": [1, 2, 3, 1, 2],
                "value": [10.0, 20.0, 30.0, 40.0, 50.0],
                "flag": ["A", "A", "B", "A", "A"],
                "__mloda_masked_src__": _USER_VALUES,
            }
        )

        result = PolarsLazyFrameAggregate._compute_frame(
            data=data,
            feature_name="sum_value",
            source_col="value",
            partition_by=["region"],
            order_by="ts",
            agg_type="sum",
            frame_type="cumulative",
            mask_spec=[("flag", "equal", "A")],
        )

        # Mask nulls row index 2 (flag == "B"); cumulative sum with forward_fill:
        # region A: [10, 30, 30 (null ffilled)], region B: [40, 90]
        assert_collision_preserved(
            result, "__mloda_masked_src__", _USER_VALUES, "sum_value", [10.0, 30.0, 30.0, 40.0, 90.0]
        )


class TestDuckdbFrameAggregateCollision:
    """User column named ``__mloda_rn__`` must survive DuckdbFrameAggregate."""

    def test_user_column_named_rn_survives(self) -> None:
        duckdb = pytest.importorskip("duckdb")
        import pyarrow as pa

        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.duckdb_frame_aggregate import (
            DuckdbFrameAggregate,
        )
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

        conn = duckdb.connect()
        arrow_table = pa.table(
            {
                "region": ["A", "A", "A", "B", "B"],
                "ts": [1, 2, 3, 1, 2],
                "value": [10.0, 20.0, 30.0, 40.0, 50.0],
                "__mloda_rn__": _USER_VALUES,
            }
        )
        rel = DuckdbRelation.from_arrow(conn, arrow_table)

        result = DuckdbFrameAggregate._compute_frame(
            data=rel,
            feature_name="sum_value",
            source_col="value",
            partition_by=["region"],
            order_by="ts",
            agg_type="sum",
            frame_type="cumulative",
        )

        assert_collision_preserved(result, "__mloda_rn__", _USER_VALUES, "sum_value", [10.0, 30.0, 60.0, 40.0, 90.0])
