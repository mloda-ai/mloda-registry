"""Helper-column collision tests for offset implementations.

These tests verify that user-provided columns whose names happen to match
internal hardcoded helper-column names survive the computation unmodified.

Helper names currently used internally:
- Pandas: ``__mloda_null_sort``
- Polars Lazy: ``__mloda_orig_idx``
- DuckDB: ``__mloda_orig_rn__``

The implementations currently clobber or drop the user column, so these
tests are expected to FAIL on the current code.
"""

from __future__ import annotations

import pytest


class TestPandasOffsetCollision:
    """User column named ``__mloda_null_sort`` must survive PandasOffset."""

    def test_user_column_named_null_sort_survives(self) -> None:
        pytest.importorskip("pandas")
        import pandas as pd

        from mloda.community.feature_groups.data_operations.row_preserving.offset.pandas_offset import (
            PandasOffset,
        )

        data = pd.DataFrame(
            {
                "region": ["A", "A", "A", "B", "B"],
                "ts": [1, 2, 3, 1, 2],
                "value": [10.0, 20.0, 30.0, 40.0, 50.0],
                "__mloda_null_sort": ["u0", "u1", "u2", "u3", "u4"],
            }
        )

        result = PandasOffset._compute_offset(
            data=data,
            feature_name="lag1_value",
            source_col="value",
            partition_by=["region"],
            order_by="ts",
            offset_type="lag_1",
        )

        assert "__mloda_null_sort" in result.columns
        assert list(result["__mloda_null_sort"]) == ["u0", "u1", "u2", "u3", "u4"]
        assert "lag1_value" in result.columns
        actual = [None if pd.isna(v) else v for v in result["lag1_value"].tolist()]
        assert actual == [None, 10.0, 20.0, None, 40.0]


class TestPolarsLazyOffsetCollision:
    """User column named ``__mloda_orig_idx`` must survive PolarsLazyOffset."""

    def test_user_column_named_orig_idx_survives(self) -> None:
        pytest.importorskip("polars")
        import polars as pl

        from mloda.community.feature_groups.data_operations.row_preserving.offset.polars_lazy_offset import (
            PolarsLazyOffset,
        )

        data = pl.LazyFrame(
            {
                "region": ["A", "A", "A", "B", "B"],
                "ts": [1, 2, 3, 1, 2],
                "value": [10.0, 20.0, 30.0, 40.0, 50.0],
                "__mloda_orig_idx": ["u0", "u1", "u2", "u3", "u4"],
            }
        )

        result = PolarsLazyOffset._compute_offset(
            data=data,
            feature_name="lag1_value",
            source_col="value",
            partition_by=["region"],
            order_by="ts",
            offset_type="lag_1",
        ).collect()

        assert "__mloda_orig_idx" in result.columns
        assert result["__mloda_orig_idx"].to_list() == ["u0", "u1", "u2", "u3", "u4"]
        assert "lag1_value" in result.columns
        assert result["lag1_value"].to_list() == [None, 10.0, 20.0, None, 40.0]


class TestDuckdbOffsetCollision:
    """User column named ``__mloda_orig_rn__`` must survive DuckdbOffset."""

    def test_user_column_named_orig_rn_survives(self) -> None:
        duckdb = pytest.importorskip("duckdb")
        import pyarrow as pa

        from mloda.community.feature_groups.data_operations.row_preserving.offset.duckdb_offset import (
            DuckdbOffset,
        )
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

        conn = duckdb.connect()
        arrow_table = pa.table(
            {
                "region": ["A", "A", "A", "B", "B"],
                "ts": [1, 2, 3, 1, 2],
                "value": [10.0, 20.0, 30.0, 40.0, 50.0],
                "__mloda_orig_rn__": ["u0", "u1", "u2", "u3", "u4"],
            }
        )
        rel = DuckdbRelation.from_arrow(conn, arrow_table)

        result = DuckdbOffset._compute_offset(
            data=rel,
            feature_name="lag1_value",
            source_col="value",
            partition_by=["region"],
            order_by="ts",
            offset_type="lag_1",
        )

        out = result.to_arrow_table()
        assert "__mloda_orig_rn__" in out.column_names
        assert out.column("__mloda_orig_rn__").to_pylist() == ["u0", "u1", "u2", "u3", "u4"]
        assert "lag1_value" in out.column_names
        assert out.column("lag1_value").to_pylist() == [None, 10.0, 20.0, None, 40.0]
