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
_MIXIN = "mloda.community.feature_groups.data_operations"


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
        """The one DuckDB mixin (shared by both families) binds the shared ``duckdb_non_numeric_descriptor``."""
        pytest.importorskip("duckdb")
        mixin = importlib.import_module(f"{_MIXIN}.duckdb_arithmetic_mixin")

        assert mixin.duckdb_non_numeric_descriptor is duckdb_numeric_source.duckdb_non_numeric_descriptor

    def test_sqlite_backends_use_shared_descriptor(self) -> None:
        """The one SQLite mixin (shared by both families) binds the shared ``sqlite_non_numeric_descriptor``."""
        mixin = importlib.import_module(f"{_MIXIN}.sqlite_arithmetic_mixin")

        assert mixin.sqlite_non_numeric_descriptor is sqlite_numeric_source.sqlite_non_numeric_descriptor

    def test_pandas_backends_use_shared_descriptor(self) -> None:
        """The one pandas mixin (shared by both families) binds the shared ``pandas_non_numeric_descriptor``."""
        pytest.importorskip("pandas")
        mixin = importlib.import_module(f"{_MIXIN}.pandas_arithmetic_mixin")

        assert mixin.pandas_non_numeric_descriptor is pandas_numeric_source.pandas_non_numeric_descriptor

    def test_polars_backends_use_shared_descriptor(self) -> None:
        """The one polars mixin (shared by both families) binds the shared ``polars_non_numeric_descriptor``."""
        pytest.importorskip("polars")
        mixin = importlib.import_module(f"{_MIXIN}.polars_arithmetic_mixin")

        assert mixin.polars_non_numeric_descriptor is polars_numeric_source.polars_non_numeric_descriptor

    def test_pyarrow_backends_use_shared_descriptor(self) -> None:
        """The one pyarrow mixin (shared by both families) binds the shared ``pyarrow_non_numeric_descriptor``."""
        pytest.importorskip("pyarrow")
        mixin = importlib.import_module(f"{_MIXIN}.pyarrow_arithmetic_mixin")

        assert mixin.pyarrow_non_numeric_descriptor is pyarrow_numeric_source.pyarrow_non_numeric_descriptor
