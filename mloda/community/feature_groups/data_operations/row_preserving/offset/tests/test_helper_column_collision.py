"""Helper-column collision tests for offset implementations.

These tests verify that user-provided columns whose names happen to match
internal hardcoded helper-column names survive the computation unmodified.

Helper names currently used internally:
- Pandas: ``__mloda_null_sort``
- Polars Lazy: ``__mloda_orig_idx``
- DuckDB: ``__mloda_orig_rn__``

Framework-agnostic assertions live in
``mloda.testing.feature_groups.data_operations.collision``.
"""

from __future__ import annotations

import pytest

from mloda.testing.feature_groups.data_operations.collision import assert_collision_preserved

_USER_VALUES = ["u0", "u1", "u2", "u3", "u4"]
_EXPECTED_LAG1 = [None, 10.0, 20.0, None, 40.0]


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
                "__mloda_null_sort": _USER_VALUES,
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

        assert_collision_preserved(result, "__mloda_null_sort", _USER_VALUES, "lag1_value", _EXPECTED_LAG1)


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
                "__mloda_orig_idx": _USER_VALUES,
            }
        )

        result = PolarsLazyOffset._compute_offset(
            data=data,
            feature_name="lag1_value",
            source_col="value",
            partition_by=["region"],
            order_by="ts",
            offset_type="lag_1",
        )

        assert_collision_preserved(result, "__mloda_orig_idx", _USER_VALUES, "lag1_value", _EXPECTED_LAG1)


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
                "__mloda_orig_rn__": _USER_VALUES,
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

        assert_collision_preserved(result, "__mloda_orig_rn__", _USER_VALUES, "lag1_value", _EXPECTED_LAG1)
