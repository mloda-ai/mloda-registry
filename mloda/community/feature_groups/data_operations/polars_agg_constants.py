"""Shared Polars aggregation-expression builders.

The aggregation and window-aggregation families share the same 14 Polars
expression builders. ``POLARS_AGG_EXPRS`` holds the shared set; the
aggregation family extends it with ``first``/``last`` (window-aggregation
implements those with deterministic ordering of its own).
"""

from __future__ import annotations

from typing import Any

import polars as pl

# Mapping from aggregation type to a Polars expression builder.
# Each callable takes a column name and returns a Polars Expr.
POLARS_AGG_EXPRS: dict[str, Any] = {
    "sum": lambda col: pl.col(col).sum(),
    "avg": lambda col: pl.col(col).mean(),
    "mean": lambda col: pl.col(col).mean(),
    "count": lambda col: pl.col(col).count(),
    "min": lambda col: pl.col(col).min(),
    "max": lambda col: pl.col(col).max(),
    "std": lambda col: pl.col(col).std(ddof=0),
    "var": lambda col: pl.col(col).var(ddof=0),
    "std_pop": lambda col: pl.col(col).std(ddof=0),
    "std_samp": lambda col: pl.col(col).std(ddof=1),
    "var_pop": lambda col: pl.col(col).var(ddof=0),
    "var_samp": lambda col: pl.col(col).var(ddof=1),
    "median": lambda col: pl.col(col).median(),
    "nunique": lambda col: pl.col(col).drop_nulls().n_unique(),
}
