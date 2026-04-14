"""Helper-column collision tests for frame_aggregate implementations.

These tests verify that user-provided columns whose names happen to match
internal hardcoded helper-column names (e.g. ``__mloda_rn__``) survive the
computation unmodified. The implementations currently clobber or drop the
user column, so these tests are expected to FAIL on the current code and
will guide the fix.
"""

from __future__ import annotations

import pytest


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
                "__mloda_rn__": ["u0", "u1", "u2", "u3", "u4"],
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

        assert "__mloda_rn__" in result.columns
        assert list(result["__mloda_rn__"]) == ["u0", "u1", "u2", "u3", "u4"]
        assert "sum_value" in result.columns
        assert list(result["sum_value"]) == [10.0, 30.0, 60.0, 40.0, 90.0]


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
                "__mloda_rn__": ["u0", "u1", "u2", "u3", "u4"],
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
        ).collect()

        assert "__mloda_rn__" in result.columns
        assert result["__mloda_rn__"].to_list() == ["u0", "u1", "u2", "u3", "u4"]
        assert "sum_value" in result.columns
        assert result["sum_value"].to_list() == [10.0, 30.0, 60.0, 40.0, 90.0]


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
                "__mloda_masked_src__": ["u0", "u1", "u2", "u3", "u4"],
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
        ).collect()

        # Mask nulls row index 2 (flag == "B"); cumulative sum with forward_fill:
        # region A: [10, 30, 30 (null ffilled)], region B: [40, 90]
        assert "__mloda_masked_src__" in result.columns
        assert result["__mloda_masked_src__"].to_list() == ["u0", "u1", "u2", "u3", "u4"]
        assert "sum_value" in result.columns
        assert result["sum_value"].to_list() == [10.0, 30.0, 30.0, 40.0, 90.0]


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
                "__mloda_rn__": ["u0", "u1", "u2", "u3", "u4"],
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

        out = result.to_arrow_table()
        assert "__mloda_rn__" in out.column_names
        assert out.column("__mloda_rn__").to_pylist() == ["u0", "u1", "u2", "u3", "u4"]
        assert "sum_value" in out.column_names
        assert out.column("sum_value").to_pylist() == [10.0, 30.0, 60.0, 40.0, 90.0]
