"""Tests for the shared numeric-source single source of truth.

These tests encode the Definition of Done for issue #214: the per-backend
"what counts as numeric" logic lives in exactly one module
(``data_operations/numeric_source.py``) rather than being duplicated across
the point- and scalar-arithmetic backend implementations.
"""

from __future__ import annotations

import pytest

from mloda.community.feature_groups.data_operations import numeric_source


class TestDuckdbNumericPrefixes:
    """The DuckDB numeric type-name allowlist lives in exactly one place."""

    def test_prefixes_attribute_exists(self) -> None:
        """``DUCKDB_NUMERIC_PREFIXES`` is exposed by the shared module."""
        assert hasattr(numeric_source, "DUCKDB_NUMERIC_PREFIXES")

    def test_prefixes_contain_expected_type_names(self) -> None:
        """The allowlist contains the expected DuckDB numeric type names."""
        prefixes = numeric_source.DUCKDB_NUMERIC_PREFIXES
        for expected in ("TINYINT", "BIGINT", "HUGEINT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "BIGNUM"):
            assert expected in prefixes

    def test_point_arithmetic_duckdb_no_longer_defines_local_prefixes(self) -> None:
        """The point-arithmetic DuckDB module must not redefine the allowlist."""
        from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic import (
            duckdb_point_arithmetic,
        )

        assert not hasattr(duckdb_point_arithmetic, "_DUCKDB_NUMERIC_PREFIXES")

    def test_scalar_arithmetic_duckdb_no_longer_defines_local_prefixes(self) -> None:
        """The scalar-arithmetic DuckDB module must not redefine the allowlist."""
        from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic import (
            duckdb_scalar_arithmetic,
        )

        assert not hasattr(duckdb_scalar_arithmetic, "_DUCKDB_NUMERIC_PREFIXES")


class TestPandasNonNumericDescriptor:
    def test_int_series_is_numeric(self) -> None:
        pd = pytest.importorskip("pandas")
        series = pd.Series([1, 2, 3])
        assert numeric_source.pandas_non_numeric_descriptor(series) is None

    def test_float_series_is_numeric(self) -> None:
        pd = pytest.importorskip("pandas")
        series = pd.Series([1.0, 2.5, 3.0])
        assert numeric_source.pandas_non_numeric_descriptor(series) is None

    def test_bool_series_is_non_numeric(self) -> None:
        pd = pytest.importorskip("pandas")
        series = pd.Series([True, False, True])
        assert numeric_source.pandas_non_numeric_descriptor(series) is not None

    def test_string_series_is_non_numeric(self) -> None:
        pd = pytest.importorskip("pandas")
        series = pd.Series(["a", "b", "c"])
        assert numeric_source.pandas_non_numeric_descriptor(series) is not None


class TestPolarsNonNumericDescriptor:
    def test_int_dtype_is_numeric(self) -> None:
        pl = pytest.importorskip("polars")
        assert numeric_source.polars_non_numeric_descriptor(pl.Int64) is None

    def test_float_dtype_is_numeric(self) -> None:
        pl = pytest.importorskip("polars")
        assert numeric_source.polars_non_numeric_descriptor(pl.Float64) is None

    def test_boolean_dtype_is_non_numeric(self) -> None:
        pl = pytest.importorskip("polars")
        assert numeric_source.polars_non_numeric_descriptor(pl.Boolean) is not None

    def test_string_dtype_is_non_numeric(self) -> None:
        pl = pytest.importorskip("polars")
        assert numeric_source.polars_non_numeric_descriptor(pl.Utf8) is not None


class TestPyarrowNonNumericDescriptor:
    def test_int_type_is_numeric(self) -> None:
        pa = pytest.importorskip("pyarrow")
        assert numeric_source.pyarrow_non_numeric_descriptor(pa.int64()) is None

    def test_float_type_is_numeric(self) -> None:
        pa = pytest.importorskip("pyarrow")
        assert numeric_source.pyarrow_non_numeric_descriptor(pa.float64()) is None

    def test_boolean_type_is_non_numeric(self) -> None:
        pa = pytest.importorskip("pyarrow")
        assert numeric_source.pyarrow_non_numeric_descriptor(pa.bool_()) is not None

    def test_string_type_is_non_numeric(self) -> None:
        pa = pytest.importorskip("pyarrow")
        assert numeric_source.pyarrow_non_numeric_descriptor(pa.string()) is not None


class TestDuckdbNonNumericDescriptor:
    def test_numeric_and_string_columns(self) -> None:
        duckdb = pytest.importorskip("duckdb")
        pa = pytest.importorskip("pyarrow")
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

        conn = duckdb.connect()
        table = pa.table({"num": pa.array([1, 2, 3], pa.int64()), "txt": pa.array(["a", "b", "c"], pa.string())})
        relation = DuckdbRelation.from_arrow(conn, table)

        assert numeric_source.duckdb_non_numeric_descriptor(relation, "num") is None

        descriptor = numeric_source.duckdb_non_numeric_descriptor(relation, "txt")
        assert descriptor is not None
        assert "VARCHAR" in str(descriptor)

    def test_absent_column_returns_none(self) -> None:
        duckdb = pytest.importorskip("duckdb")
        pa = pytest.importorskip("pyarrow")
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

        conn = duckdb.connect()
        table = pa.table({"num": pa.array([1, 2, 3], pa.int64())})
        relation = DuckdbRelation.from_arrow(conn, table)

        assert numeric_source.duckdb_non_numeric_descriptor(relation, "missing") is None


class TestSqliteNonNumericDescriptor:
    def test_numeric_and_string_columns(self) -> None:
        pytest.importorskip("pyarrow")
        import sqlite3

        import pyarrow as pa

        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

        conn = sqlite3.connect(":memory:")
        table = pa.table({"num": pa.array([1, 2, 3], pa.int64()), "txt": pa.array(["a", "b", "c"], pa.string())})
        relation = SqliteRelation.from_arrow(conn, table)

        assert numeric_source.sqlite_non_numeric_descriptor(relation, "num") is None

        descriptor = numeric_source.sqlite_non_numeric_descriptor(relation, "txt")
        assert descriptor is not None
        assert "affinity" in str(descriptor)

    def test_absent_column_returns_none(self) -> None:
        pytest.importorskip("pyarrow")
        import sqlite3

        import pyarrow as pa

        from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

        conn = sqlite3.connect(":memory:")
        table = pa.table({"num": pa.array([1, 2, 3], pa.int64())})
        relation = SqliteRelation.from_arrow(conn, table)

        assert numeric_source.sqlite_non_numeric_descriptor(relation, "missing") is None
