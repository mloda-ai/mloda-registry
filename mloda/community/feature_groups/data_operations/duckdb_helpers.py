"""Shared DuckDB helper utilities for time bucketization, resample, and aggregation.

Centralizes the epoch-anchored floor expression (and the interval-literal
building block it depends on) so every DuckDB-based bucket/resample feature
group floors timestamps identically, plus a column-type lookup and a
float-only NaN-to-null wrap shared by aggregation feature groups.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

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


def column_types(data: "DuckdbRelation") -> dict[str, str]:
    """Column name -> DuckDB type string, read from the relation's schema."""
    return dict(zip(data.columns, [str(t) for t in data.types]))


def nan_to_null_sql(expr: str, column_type: str) -> str:
    """Wrap ``expr`` so NaN counts as null in aggregate functions, float columns only."""
    if column_type.upper() in ("FLOAT", "DOUBLE"):
        return f"NULLIF({expr}, 'NaN')"
    return expr


def nan_policy_agg_sql(data: "DuckdbRelation", source_col: str, source_sql: str, agg_type: str, agg_func: str) -> str:
    """Full aggregate call for ``agg_type``, applying the NaN policy natively.

    ``mode`` returns a struct, so callers must extract ``.v``. Float ``max`` becomes
    ``-MIN(-x)``: DuckDB sorts NaN highest, so MIN skips it and an all-NaN input stays NaN.
    """
    if agg_type == "mode":
        return f"MODE(CASE WHEN {source_sql} IS NOT NULL THEN struct_pack(v := {source_sql}) END)"

    column_type = column_types(data).get(source_col, "")
    is_float = column_type.upper() in ("FLOAT", "DOUBLE")

    if is_float and agg_type == "median":
        return f"{agg_func}({nan_to_null_sql(source_sql, column_type)})"
    if is_float and agg_type == "max":
        return f"-MIN(-({source_sql}))"
    return f"{agg_func}({source_sql})"
