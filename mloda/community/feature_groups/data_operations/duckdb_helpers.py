"""Shared DuckDB helper utilities for time bucketization and resample.

Centralizes the epoch-anchored floor expression (and the interval-literal
building block it depends on) so every DuckDB-based bucket/resample feature
group floors timestamps identically.
"""

from __future__ import annotations

# DuckDB ``DATE_TRUNC`` unit names per logical unit.
DUCKDB_TRUNC_UNIT: dict[str, str] = {
    "minute": "minute",
    "hour": "hour",
    "day": "day",
    "week": "week",
    "month": "month",
    "year": "year",
}


def interval_literal(n: int, unit: str) -> str:
    """DuckDB interval literal for ``n`` units of ``unit``."""
    if unit == "minute":
        return f"INTERVAL {n} MINUTE"
    if unit == "hour":
        return f"INTERVAL {n} HOUR"
    if unit == "day":
        return f"INTERVAL {n} DAY"
    if unit == "week":
        return "INTERVAL 1 WEEK"
    if unit == "month":
        return "INTERVAL 1 MONTH"
    if unit == "year":
        return "INTERVAL 1 YEAR"
    raise ValueError(f"Unsupported time bucketization unit for DuckDB: {unit!r}")


def floor_expr(quoted_source: str, n: int, unit: str) -> str:
    """SQL flooring a DuckDB timestamp to the ``(n, unit)`` bucket.

    Shared entry point for both DuckDB backends. Requires a UTC session tz,
    guaranteed by ``DuckDBFramework`` (mloda >= 0.9.0); do not add a local pin.
    """
    if n == 1:
        return f"DATE_TRUNC('{DUCKDB_TRUNC_UNIT[unit]}', {quoted_source})"
    # ``n > 1`` is only valid for fixed-freq units (minute/hour/day).
    interval = interval_literal(n, unit)
    # Pin the origin to 1970-01-01 to match PyArrow's bucket alignment
    # (multiples since the epoch). Without an explicit origin, DuckDB's
    # ``time_bucket`` anchors sub-month widths at 2000-01-03, which
    # diverges from PyArrow on multi-day buckets. DATE auto-casts to both
    # TIMESTAMP and TIMESTAMPTZ so the same literal works for either
    # source column type.
    return f"time_bucket({interval}, {quoted_source}, DATE '1970-01-01')"
