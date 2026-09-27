"""Shared Polars helpers for time bucketization, resample, and aggregation.

Centralizes the unit-alias table and the ``(n, unit)`` -> duration string
formatting so every Polars-based bucket/resample feature group builds the
same ``dt.truncate`` / ``dt.round`` tokens, plus a float-only NaN-to-null
wrap shared by aggregation feature groups.
"""

from __future__ import annotations

from typing import Callable

import polars as pl

# Polars duration aliases for each unit. Polars' ``dt.truncate('1w')`` is
# Monday-anchored, which matches the ISO week convention pinned by the FG.
POLARS_UNIT_ALIASES: dict[str, str] = {
    "minute": "m",
    "hour": "h",
    "day": "d",
    "week": "w",
    "month": "mo",
    "year": "y",
}


def duration_token(n: int, unit: str) -> str:
    """Format the Polars duration token for ``(n, unit)`` (e.g. ``5m``, ``1d``)."""
    return f"{n}{POLARS_UNIT_ALIASES[unit]}"


def nan_to_null(expr: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    """Fill NaN with null so NaN counts as null in aggregations, float dtypes only."""
    if dtype.is_float():
        return expr.fill_nan(None)
    return expr


def nan_skipping_extreme(expr_fn: Callable[[pl.Expr], pl.Expr], col: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    """``expr_fn(col)`` with NaN skipped like pc.min/pc.max, an all-NaN window staying NaN.

    Float dtypes run ``expr_fn`` on the NaN-to-null column, then fall back to
    ``expr_fn(col)`` (where NaN propagates) for null results, only true if the window was all-NaN or all-null.
    """
    if not dtype.is_float():
        return expr_fn(col)
    return expr_fn(nan_to_null(col, dtype)).fill_null(expr_fn(col))
