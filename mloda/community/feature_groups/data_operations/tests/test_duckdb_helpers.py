"""Tests for the DuckDB float-only NaN-to-null wrap helper."""

from __future__ import annotations

from mloda.community.feature_groups.data_operations.duckdb_helpers import nan_to_null_sql


class TestNanToNullSql:
    def test_float_type_wraps_with_nullif(self) -> None:
        """FLOAT/DOUBLE columns get NULLIF(expr, 'NaN'), so NaN counts as null in aggregate functions."""
        assert nan_to_null_sql("val", "DOUBLE") == "NULLIF(val, 'NaN')"
        assert nan_to_null_sql("val", "FLOAT") == "NULLIF(val, 'NaN')"

    def test_non_float_type_passes_through_unchanged(self) -> None:
        """A non-float column (e.g. TIMESTAMP, BIGINT) is returned unchanged."""
        assert nan_to_null_sql("ts", "TIMESTAMP") == "ts"
        assert nan_to_null_sql("n", "BIGINT") == "n"
