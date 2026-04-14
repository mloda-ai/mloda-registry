"""Helper-column collision tests for window_aggregation implementations.

These tests verify that user-provided columns whose names happen to match
internal hardcoded helper-column names survive the computation unmodified.

Helper names currently used internally:
- PyArrow: ``__mloda_wa_idx__``
- DuckDB: ``__mloda_rn__``
- Reference: ``__mloda_wa_idx__``
"""

from __future__ import annotations

import pytest


class TestPyArrowWindowCollision:
    """User column named ``__mloda_wa_idx__`` must survive PyArrowWindowAggregation."""

    def test_user_column_named_wa_idx_survives(self) -> None:
        pa = pytest.importorskip("pyarrow")

        from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.pyarrow_window_aggregation import (
            PyArrowWindowAggregation,
        )

        table = pa.table(
            {
                "region": ["A", "A", "A", "B", "B"],
                "value": [10.0, 20.0, 30.0, 40.0, 50.0],
                "__mloda_wa_idx__": ["u0", "u1", "u2", "u3", "u4"],
            }
        )

        result = PyArrowWindowAggregation._compute_window(
            table=table,
            feature_name="sum_value",
            source_col="value",
            partition_by=["region"],
            agg_type="sum",
        )

        assert "__mloda_wa_idx__" in result.column_names
        assert result.column("__mloda_wa_idx__").to_pylist() == ["u0", "u1", "u2", "u3", "u4"]
        assert "sum_value" in result.column_names
        assert result.column("sum_value").to_pylist() == [60.0, 60.0, 60.0, 90.0, 90.0]


class TestDuckdbWindowCollision:
    """User column named ``__mloda_rn__`` must survive DuckdbWindowAggregation.

    The helper column is only used by the first/last paths, so this test
    uses ``agg_type="first"`` to exercise that path.
    """

    def test_user_column_named_rn_survives_in_first(self) -> None:
        duckdb = pytest.importorskip("duckdb")
        import pyarrow as pa

        from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.duckdb_window_aggregation import (
            DuckdbWindowAggregation,
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

        result = DuckdbWindowAggregation._compute_window(
            data=rel,
            feature_name="first_value",
            source_col="value",
            partition_by=["region"],
            agg_type="first",
            order_by="ts",
        )

        out = result.to_arrow_table()
        assert "__mloda_rn__" in out.column_names
        assert out.column("__mloda_rn__").to_pylist() == ["u0", "u1", "u2", "u3", "u4"]
        assert "first_value" in out.column_names
        assert out.column("first_value").to_pylist() == [10.0, 10.0, 10.0, 40.0, 40.0]


class TestPolarsLazyWindowAggregationMaskCollision:
    """User column named ``__mloda_masked_src__`` must survive PolarsLazyWindowAggregation."""

    def test_user_column_named_masked_src_survives(self) -> None:
        pytest.importorskip("polars")
        import polars as pl

        from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.polars_lazy_window_aggregation import (
            PolarsLazyWindowAggregation,
        )

        data = pl.LazyFrame(
            {
                "region": ["A", "A", "A", "B", "B"],
                "value": [10.0, 20.0, 30.0, 40.0, 50.0],
                "flag": ["A", "A", "B", "A", "A"],
                "__mloda_masked_src__": ["u0", "u1", "u2", "u3", "u4"],
            }
        )

        result = PolarsLazyWindowAggregation._compute_window(
            data=data,
            feature_name="sum_value",
            source_col="value",
            partition_by=["region"],
            agg_type="sum",
            mask_spec=[("flag", "equal", "A")],
        ).collect()

        # Mask keeps flag == "A": region A -> 10+20 = 30, region B -> 40+50 = 90
        assert "__mloda_masked_src__" in result.columns
        assert result["__mloda_masked_src__"].to_list() == ["u0", "u1", "u2", "u3", "u4"]
        assert "sum_value" in result.columns
        assert result["sum_value"].to_list() == [30.0, 30.0, 30.0, 90.0, 90.0]


class TestReferenceWindowCollision:
    """User column named ``__mloda_wa_idx__`` must survive ReferenceWindowAggregation."""

    def test_user_column_named_wa_idx_survives(self) -> None:
        pa = pytest.importorskip("pyarrow")

        from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation.reference import (
            ReferenceWindowAggregation,
        )

        table = pa.table(
            {
                "region": ["A", "A", "A", "B", "B"],
                "value": [10.0, 20.0, 30.0, 40.0, 50.0],
                "__mloda_wa_idx__": ["u0", "u1", "u2", "u3", "u4"],
            }
        )

        result = ReferenceWindowAggregation._compute_window(
            table=table,
            feature_name="sum_value",
            source_col="value",
            partition_by=["region"],
            agg_type="sum",
        )

        assert "__mloda_wa_idx__" in result.column_names
        assert result.column("__mloda_wa_idx__").to_pylist() == ["u0", "u1", "u2", "u3", "u4"]
        assert "sum_value" in result.column_names
        assert result.column("sum_value").to_pylist() == [60.0, 60.0, 60.0, 90.0, 90.0]
