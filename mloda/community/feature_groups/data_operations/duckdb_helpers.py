"""Shared DuckDB helper utilities for time bucketization, resample, and aggregation.

Centralizes the epoch-anchored floor expression (and the interval-literal
building block it depends on) so every DuckDB-based bucket/resample feature
group floors timestamps identically, plus a column-type lookup and a
float-only NaN-to-null wrap shared by aggregation feature groups.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

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


_FLOAT_TYPES = ("FLOAT", "DOUBLE")


def _is_float(data: "DuckdbRelation", source_col: str) -> bool:
    """Whether ``source_col`` is a DuckDB FLOAT/DOUBLE column."""
    return column_types(data).get(source_col, "").upper() in _FLOAT_TYPES


def nan_to_null_sql(expr: str, column_type: str) -> str:
    """Wrap ``expr`` so NaN counts as null in aggregate functions, float columns only."""
    if column_type.upper() in _FLOAT_TYPES:
        return f"NULLIF({expr}, 'NaN')"
    return expr


# std/var agg_type -> the COVAR_POP/COVAR_SAMP call it maps to on float columns.
# STDDEV_*/VAR_* raise OutOfRangeException on NaN input; COVAR_*(x, x) equals VAR_*(x)
# but propagates NaN instead of raising.
_FLOAT_VAR_COVAR: dict[str, str] = {"var": "COVAR_POP", "var_pop": "COVAR_POP", "var_samp": "COVAR_SAMP"}
# std agg_type -> the var agg_type it reduces to, before being wrapped in SQRT.
_FLOAT_STD_TO_VAR: dict[str, str] = {"std": "var_pop", "std_pop": "var_pop", "std_samp": "var_samp"}


def nan_policy_agg_sql(data: "DuckdbRelation", source_col: str, source_sql: str, agg_type: str, agg_func: str) -> str:
    """Full aggregate call for ``agg_type``, applying the NaN policy natively.

    ``mode`` returns a struct, so callers must extract ``.v``. Float ``max`` becomes
    ``-MIN(-x)``: DuckDB sorts NaN highest, so MIN skips it and an all-NaN input stays NaN.
    Float std/var become ``COVAR_POP``/``COVAR_SAMP``, which propagate NaN instead of raising.
    Callers using ``window()`` must go through ``nan_policy_window`` instead: float std here
    returns ``SQRT(...)``, which cannot take an ``OVER`` clause.
    """
    if agg_type == "mode":
        return f"MODE(CASE WHEN {source_sql} IS NOT NULL THEN struct_pack(v := {source_sql}) END)"

    is_float = _is_float(data, source_col)

    if is_float and agg_type == "median":
        return f"{agg_func}({nan_to_null_sql(source_sql, column_types(data).get(source_col, ''))})"
    if is_float and agg_type == "max":
        return f"-MIN(-({source_sql}))"
    if is_float and agg_type in _FLOAT_STD_TO_VAR:
        return f"SQRT({nan_policy_agg_sql(data, source_col, source_sql, _FLOAT_STD_TO_VAR[agg_type], agg_func)})"
    if is_float and agg_type in _FLOAT_VAR_COVAR:
        return f"{_FLOAT_VAR_COVAR[agg_type]}({source_sql}, {source_sql})"
    return f"{agg_func}({source_sql})"


def nan_policy_window(
    data: "DuckdbRelation",
    source_col: str,
    source_sql: str,
    agg_type: str,
    agg_func: str,
    feature_name: str,
    **window_kwargs: Any,
) -> "DuckdbRelation":
    """``data.window(...)`` applying the NaN policy.

    ``window()`` takes a single aggregate call, so std passes the variance call in and
    wraps the resulting column in ``SQRT`` afterward.
    """
    var_type = _FLOAT_STD_TO_VAR.get(agg_type) if _is_float(data, source_col) else None
    agg_call = nan_policy_agg_sql(data, source_col, source_sql, var_type or agg_type, agg_func)
    result = data.window(agg_call, feature_name, **window_kwargs)

    if var_type is None:
        return result

    quoted_feature = quote_ident(feature_name)
    keep = ", ".join(
        f"SQRT({quoted_feature}) AS {quoted_feature}" if c == feature_name else quote_ident(c) for c in result.columns
    )
    return result.project(keep)
