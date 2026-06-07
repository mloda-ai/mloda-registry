"""Shared DuckDB helper utilities for the data-operations backends."""

from __future__ import annotations

from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation


def assert_source_col_present(data: DuckdbRelation, col: str) -> None:
    """Reject a missing source column on a DuckDB relation with a clear ``ValueError``."""
    if col not in data.columns:
        raise ValueError(f"Source column {col!r} is not present in the DuckDB relation; available: {data.columns}.")
