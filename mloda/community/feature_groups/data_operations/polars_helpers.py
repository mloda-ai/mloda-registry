"""Shared Polars helper utilities for row-preserving data operations."""

from __future__ import annotations

from typing import Any


def assert_polars_source_col_present(data: Any, source_col: str) -> None:
    """Reject a missing source column on a Polars LazyFrame with a clear ``ValueError``."""
    names = data.collect_schema().names()
    if source_col not in names:
        raise ValueError(f"Source column {source_col!r} is not present in the polars frame; available: {names}.")
