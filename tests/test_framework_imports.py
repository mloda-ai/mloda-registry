"""Verify that compute-framework packages required by tests are installed.

These tests fail loudly (instead of silently skipping) when a framework
dependency is missing from the dev extras in the root pyproject.toml.
See: https://github.com/mloda-ai/mloda-registry/issues/136
"""

from __future__ import annotations


def test_polars_importable() -> None:
    """polars must be installed in the test environment."""
    import polars  # noqa: F401


def test_pandas_importable() -> None:
    """pandas must be installed in the test environment."""
    import pandas  # noqa: F401


def test_duckdb_importable() -> None:
    """duckdb must be installed in the test environment."""
    import duckdb  # noqa: F401


def test_pyarrow_importable() -> None:
    """pyarrow must be installed in the test environment (via mloda)."""
    import pyarrow  # noqa: F401
