"""Shared polars helper utilities for the data-operations backends."""

from __future__ import annotations

import polars as pl


def assert_source_col_present(data: pl.LazyFrame, col: str) -> None:
    """Reject a missing source column on a polars LazyFrame with a clear ``ValueError``."""
    names = data.collect_schema().names()
    if col not in names:
        raise ValueError(f"Source column {col!r} is not present in the polars frame; available: {names}.")
