"""Helper-column collision tests for binning implementations.

These tests verify that user-provided columns whose names happen to match
internal hardcoded helper-column names survive the computation unmodified.

DuckDB binning uses a ``__mloda_rn__`` helper column in its ``qbin`` path.
"""

from __future__ import annotations

import pytest


class TestDuckdbBinningCollision:
    """User column named ``__mloda_rn__`` must survive DuckdbBinning qbin."""

    def test_user_column_named_rn_survives_in_qbin(self) -> None:
        duckdb = pytest.importorskip("duckdb")
        import pyarrow as pa

        from mloda.community.feature_groups.data_operations.row_preserving.binning.duckdb_binning import (
            DuckdbBinning,
        )
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

        conn = duckdb.connect()
        arrow_table = pa.table(
            {
                "value": [1.0, 2.0, 3.0, 4.0],
                "__mloda_rn__": ["u0", "u1", "u2", "u3"],
            }
        )
        rel = DuckdbRelation.from_arrow(conn, arrow_table)

        result = DuckdbBinning._compute_binning(
            data=rel,
            feature_name="qbin_value",
            source_col="value",
            op="qbin",
            n_bins=2,
        )

        out = result.to_arrow_table()
        assert "__mloda_rn__" in out.column_names
        assert out.column("__mloda_rn__").to_pylist() == ["u0", "u1", "u2", "u3"]
        assert "qbin_value" in out.column_names
        assert out.column("qbin_value").to_pylist() == [0, 0, 1, 1]
