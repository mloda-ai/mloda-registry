"""Shared DuckDB helper utilities for row-preserving data operations."""

from __future__ import annotations

from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident


def assert_duckdb_source_col_present(data: DuckdbRelation, source_col: str) -> None:
    """Reject a missing source column on a DuckDB relation with a clear ``ValueError``."""
    if source_col not in data.columns:
        raise ValueError(
            f"Source column {source_col!r} is not present in the DuckDB relation; available: {data.columns}."
        )


def duckdb_drop_rn_restore(rel: DuckdbRelation, rn: str) -> DuckdbRelation:
    """Restore original row order via the ``rn`` helper column, then drop it.

    Orders the relation by the row-number helper column *rn* (recreating the
    original input order after a window/aggregate reshuffle), projects every
    remaining column, and drops *rn*.
    """
    rel = rel.order(quote_ident(rn))
    keep = ", ".join(quote_ident(c) for c in rel.columns if c != rn)
    return rel.project(keep)
