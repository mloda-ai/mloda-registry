"""Helper-column collision tests for rank implementations.

These tests verify that user-provided columns whose names happen to match
internal hardcoded helper-column names survive the computation unmodified.

Helper names currently used internally:
- Polars Lazy: ``__mloda_rank_null_flag__``
- DuckDB: ``__mloda_orig_rn``
"""

from __future__ import annotations

import pytest


class TestPolarsLazyRankCollision:
    """User column named ``__mloda_rank_null_flag__`` must survive PolarsLazyRank."""

    def test_user_column_named_null_flag_survives(self) -> None:
        pytest.importorskip("polars")
        import polars as pl

        from mloda.community.feature_groups.data_operations.row_preserving.rank.polars_lazy_rank import (
            PolarsLazyRank,
        )

        data = pl.LazyFrame(
            {
                "region": ["A", "A", "A", "B", "B"],
                "ts": [1, 2, 3, 1, 2],
                "__mloda_rank_null_flag__": ["u0", "u1", "u2", "u3", "u4"],
            }
        )

        result = PolarsLazyRank._compute_rank(
            data=data,
            feature_name="rn",
            partition_by=["region"],
            order_by="ts",
            rank_type="row_number",
        ).collect()

        assert "__mloda_rank_null_flag__" in result.columns
        assert result["__mloda_rank_null_flag__"].to_list() == ["u0", "u1", "u2", "u3", "u4"]
        assert "rn" in result.columns
        assert result["rn"].to_list() == [1, 2, 3, 1, 2]


class TestDuckdbRankCollision:
    """User column named ``__mloda_orig_rn`` must survive DuckdbRank."""

    def test_user_column_named_orig_rn_survives(self) -> None:
        duckdb = pytest.importorskip("duckdb")
        import pyarrow as pa

        from mloda.community.feature_groups.data_operations.row_preserving.rank.duckdb_rank import (
            DuckdbRank,
        )
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

        conn = duckdb.connect()
        arrow_table = pa.table(
            {
                "region": ["A", "A", "A", "B", "B"],
                "ts": [1, 2, 3, 1, 2],
                "__mloda_orig_rn": ["u0", "u1", "u2", "u3", "u4"],
            }
        )
        rel = DuckdbRelation.from_arrow(conn, arrow_table)

        result = DuckdbRank._compute_rank(
            data=rel,
            feature_name="rn",
            partition_by=["region"],
            order_by="ts",
            rank_type="row_number",
        )

        out = result.to_arrow_table()
        assert "__mloda_orig_rn" in out.column_names
        assert out.column("__mloda_orig_rn").to_pylist() == ["u0", "u1", "u2", "u3", "u4"]
        assert "rn" in out.column_names
        assert out.column("rn").to_pylist() == [1, 2, 3, 1, 2]
