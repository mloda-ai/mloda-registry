"""Structural single-source-of-truth guards for the per-framework numeric-source modules.

These tests guard the "exactly one place" Definition of Done for issue #214:
the per-backend "what counts as numeric" logic lives in exactly one module per
compute framework (``data_operations/<framework>_numeric_source.py``) and each
such module is imported by BOTH the point- and scalar-arithmetic families from
that single per-framework source.

They are deliberately structural (constant content, no module-level
redeclaration, ``is``-identity of the imported descriptor functions). The actual
per-backend numeric behavior on real data is covered by the shared
``PointArithmeticTestBase`` / ``ScalarArithmeticTestBase`` non-numeric source
rejection tests, which run across all five backends on the canonical 12-row
dataset and flow through the per-framework numeric-source modules after the refactor.
"""

from __future__ import annotations

import importlib

import pytest

from mloda.community.feature_groups.data_operations import (
    duckdb_numeric_source,
    pandas_numeric_source,
    polars_numeric_source,
    pyarrow_numeric_source,
    sqlite_numeric_source,
)

_POINT = "mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic"
_SCALAR = "mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic"


class TestNumericSourceSingleSourceOfTruth:
    """Guard that each framework's numeric-source logic lives in one module, imported by both families."""

    def test_duckdb_numeric_prefixes_content(self) -> None:
        """The DuckDB numeric allowlist is exposed by the shared module with the expected names."""
        prefixes = duckdb_numeric_source.DUCKDB_NUMERIC_PREFIXES
        for expected in ("TINYINT", "BIGINT", "HUGEINT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "BIGNUM"):
            assert expected in prefixes, f"{expected!r} missing from duckdb_numeric_source.DUCKDB_NUMERIC_PREFIXES"

    def test_duckdb_backend_modules_do_not_redeclare_prefixes(self) -> None:
        """Neither DuckDB backend may carry its own copy of the numeric allowlist."""
        pytest.importorskip("duckdb")
        for path in (
            f"{_POINT}.duckdb_point_arithmetic",
            f"{_SCALAR}.duckdb_scalar_arithmetic",
        ):
            module = importlib.import_module(path)
            assert not hasattr(module, "_DUCKDB_NUMERIC_PREFIXES"), (
                f"{path} still declares a module-level _DUCKDB_NUMERIC_PREFIXES; "
                "it must be removed in favour of duckdb_numeric_source.DUCKDB_NUMERIC_PREFIXES."
            )

    def test_duckdb_backends_use_shared_descriptor(self) -> None:
        """Both DuckDB backends bind the one shared ``duckdb_non_numeric_descriptor``."""
        pytest.importorskip("duckdb")
        point = importlib.import_module(f"{_POINT}.duckdb_point_arithmetic")
        scalar = importlib.import_module(f"{_SCALAR}.duckdb_scalar_arithmetic")

        assert point.duckdb_non_numeric_descriptor is duckdb_numeric_source.duckdb_non_numeric_descriptor
        assert scalar.duckdb_non_numeric_descriptor is duckdb_numeric_source.duckdb_non_numeric_descriptor

    def test_sqlite_backends_use_shared_descriptor(self) -> None:
        """Both SQLite backends bind the one shared ``sqlite_non_numeric_descriptor``."""
        point = importlib.import_module(f"{_POINT}.sqlite_point_arithmetic")
        scalar = importlib.import_module(f"{_SCALAR}.sqlite_scalar_arithmetic")

        assert point.sqlite_non_numeric_descriptor is sqlite_numeric_source.sqlite_non_numeric_descriptor
        assert scalar.sqlite_non_numeric_descriptor is sqlite_numeric_source.sqlite_non_numeric_descriptor

    def test_pandas_backends_use_shared_descriptor(self) -> None:
        """Both pandas backends bind the one shared ``pandas_non_numeric_descriptor``."""
        pytest.importorskip("pandas")
        point = importlib.import_module(f"{_POINT}.pandas_point_arithmetic")
        scalar = importlib.import_module(f"{_SCALAR}.pandas_scalar_arithmetic")

        assert point.pandas_non_numeric_descriptor is pandas_numeric_source.pandas_non_numeric_descriptor
        assert scalar.pandas_non_numeric_descriptor is pandas_numeric_source.pandas_non_numeric_descriptor

    def test_polars_backends_use_shared_descriptor(self) -> None:
        """Both polars backends bind the one shared ``polars_non_numeric_descriptor``."""
        pytest.importorskip("polars")
        point = importlib.import_module(f"{_POINT}.polars_lazy_point_arithmetic")
        scalar = importlib.import_module(f"{_SCALAR}.polars_lazy_scalar_arithmetic")

        assert point.polars_non_numeric_descriptor is polars_numeric_source.polars_non_numeric_descriptor
        assert scalar.polars_non_numeric_descriptor is polars_numeric_source.polars_non_numeric_descriptor

    def test_pyarrow_backends_use_shared_descriptor(self) -> None:
        """Both pyarrow backends bind the one shared ``pyarrow_non_numeric_descriptor``."""
        pytest.importorskip("pyarrow")
        point = importlib.import_module(f"{_POINT}.pyarrow_point_arithmetic")
        scalar = importlib.import_module(f"{_SCALAR}.pyarrow_scalar_arithmetic")

        assert point.pyarrow_non_numeric_descriptor is pyarrow_numeric_source.pyarrow_non_numeric_descriptor
        assert scalar.pyarrow_non_numeric_descriptor is pyarrow_numeric_source.pyarrow_non_numeric_descriptor
