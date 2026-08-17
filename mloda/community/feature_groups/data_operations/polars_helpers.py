"""Shared Polars duration-token helper for time bucketization and resample.

Centralizes the unit-alias table and the ``(n, unit)`` -> duration string
formatting so every Polars-based bucket/resample feature group builds the
same ``dt.truncate`` / ``dt.round`` tokens.
"""

from __future__ import annotations

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
