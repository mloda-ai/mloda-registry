"""Helper-column collision tests for rank implementations.

These tests verify that user-provided columns whose names happen to match
internal hardcoded helper-column names survive the computation unmodified.

Helper names currently used internally:
- Polars Lazy: ``__mloda_rank_null_flag__``
- DuckDB: ``__mloda_orig_rn``

Framework-agnostic assertions live in
``mloda.testing.feature_groups.data_operations.collision``.
"""

from __future__ import annotations

import pytest

from mloda.testing.feature_groups.data_operations.collision import assert_collision_preserved

_USER_VALUES = ["u0", "u1", "u2", "u3", "u4"]
_EXPECTED_RN = [1, 2, 3, 1, 2]


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
                "__mloda_rank_null_flag__": _USER_VALUES,
            }
        )

        result = PolarsLazyRank._compute_rank(
            data=data,
            feature_name="rn",
            partition_by=["region"],
            order_by="ts",
            rank_type="row_number",
        )

        assert_collision_preserved(result, "__mloda_rank_null_flag__", _USER_VALUES, "rn", _EXPECTED_RN)


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
                "__mloda_orig_rn": _USER_VALUES,
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

        assert_collision_preserved(result, "__mloda_orig_rn", _USER_VALUES, "rn", _EXPECTED_RN)
