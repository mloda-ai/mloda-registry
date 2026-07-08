"""DuckDB implementation of time bucketization.

Floors use ``DATE_TRUNC`` for calendar/single-unit cases (Mon-anchored for
``week``, verified) and ``time_bucket`` for fixed-freq ``n > 1`` cases.
Ceil uses CASE-based projection that matches PyArrow's
``ceil_temporal(ceil_is_strictly_greater=False)`` per-unit quirk: idempotent
on aligned for fixed-freq, always-advancing for calendar units. Round is
half-up via midpoint comparison on epoch seconds (matches PyArrow's
``round_temporal`` default).
"""

from __future__ import annotations

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_framework import DuckDBFramework
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident

from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.base import (
    TIME_BUCKETIZATION_OPS,
    TimeBucketizationFeatureGroup,
)

# Calendar units whose ``ceil`` always advances by one bucket even on
# aligned input (matches PyArrow's ``ceil_temporal`` quirk for
# ``week`` / ``month`` / ``year``).
_CALENDAR_CEIL_ALWAYS_ADVANCES: frozenset[str] = frozenset({"week", "month", "year"})

# DuckDB ``DATE_TRUNC`` unit names per logical unit.
_DUCKDB_TRUNC_UNIT: dict[str, str] = {
    "minute": "minute",
    "hour": "hour",
    "day": "day",
    "week": "week",
    "month": "month",
    "year": "year",
}

# DuckDB type names that imply a timestamp source. Parameterized variants
# (``TIMESTAMP_NS``, ``TIMESTAMP WITH TIME ZONE``) are matched as prefixes.
# Bare DATE is intentionally NOT accepted: round at sub-day units fails
# inside the SQL with a cryptic BinderException about epoch(BIGINT). Users
# with DATE columns should cast to TIMESTAMP before bucketing.
_DUCKDB_TIMESTAMP_PREFIXES: tuple[str, ...] = (
    "TIMESTAMP",
    "DATETIME",
)


def _interval_literal(n: int, unit: str) -> str:
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
        return f"DATE_TRUNC('{_DUCKDB_TRUNC_UNIT[unit]}', {quoted_source})"
    # ``n > 1`` is only valid for fixed-freq units (minute/hour/day).
    interval = _interval_literal(n, unit)
    # Pin the origin to 1970-01-01 to match PyArrow's bucket alignment
    # (multiples since the epoch). Without an explicit origin, DuckDB's
    # ``time_bucket`` anchors sub-month widths at 2000-01-03, which
    # diverges from PyArrow on multi-day buckets. DATE auto-casts to both
    # TIMESTAMP and TIMESTAMPTZ so the same literal works for either
    # source column type.
    return f"time_bucket({interval}, {quoted_source}, DATE '1970-01-01')"


def _bucket_seconds_for_fixed_freq(n: int, unit: str) -> int:
    """Length of one ``(n, unit)`` bucket in seconds (fixed-freq only)."""
    if unit == "minute":
        return 60 * n
    if unit == "hour":
        return 3600 * n
    if unit == "day":
        return 86400 * n
    raise ValueError(f"Bucket-seconds helper called for non-fixed-freq unit: {unit!r}")


class DuckdbTimeBucketization(TimeBucketizationFeatureGroup):
    """DuckDB backend for time bucketization."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {DuckDBFramework}

    @classmethod
    def _assert_source_column_is_timestamp(cls, data: DuckdbRelation, source_col: str) -> None:
        underlying = data._relation
        type_by_column = dict(zip(list(underlying.columns), [str(t) for t in underlying.types]))
        dtype_str = type_by_column.get(source_col)
        if dtype_str is None:
            raise ValueError(
                f"Source column {source_col!r} is not present in the DuckDB relation; "
                f"available: {list(type_by_column)}."
            )
        upper = dtype_str.upper()
        if not any(upper == p or upper.startswith(p) for p in _DUCKDB_TIMESTAMP_PREFIXES):
            cls._raise_non_timestamp_source(source_col, dtype_str)

    @classmethod
    def _compute_bucket(
        cls,
        data: DuckdbRelation,
        feature_name: str,
        source_col: str,
        op: str,
        n: int,
        unit: str,
    ) -> DuckdbRelation:
        if op not in TIME_BUCKETIZATION_OPS:
            raise ValueError(f"Unsupported bucket op {op!r} for DuckDB; supported: {sorted(TIME_BUCKETIZATION_OPS)}.")

        quoted_source = quote_ident(source_col)
        quoted_feature = quote_ident(feature_name)
        floor_sql = floor_expr(quoted_source, n, unit)
        interval = _interval_literal(n, unit)
        ceil_next = f"({floor_sql} + {interval})"

        if op == "floor":
            result_expr = floor_sql
        elif op == "ceil":
            if unit in _CALENDAR_CEIL_ALWAYS_ADVANCES:
                # Calendar units always advance.
                result_expr = f"CASE WHEN {quoted_source} IS NULL THEN NULL ELSE {ceil_next} END"
            else:
                result_expr = (
                    f"CASE "
                    f"WHEN {quoted_source} IS NULL THEN NULL "
                    f"WHEN {quoted_source} = {floor_sql} THEN {floor_sql} "
                    f"ELSE {ceil_next} "
                    f"END"
                )
        else:  # round
            result_expr = cls._round_expression(quoted_source, n, unit, floor_sql, ceil_next)

        raw_sql = f"*, {result_expr} AS {quoted_feature}"
        result: DuckdbRelation = data.project(raw_sql)
        return result

    @classmethod
    def _round_expression(
        cls,
        quoted_source: str,
        n: int,
        unit: str,
        floor_sql: str,
        ceil_next: str,
    ) -> str:
        """SQL expression for round-half-up bucketization.

        Uses ``EPOCH(col - floored) >= bucket_seconds / 2`` for fixed-freq
        units, and ``EPOCH(col - floored) * 2 >= EPOCH(ceil_next - floored)``
        for calendar units (whose bucket lengths vary per row).
        """
        offset_seconds = f"EPOCH({quoted_source} - {floor_sql})"
        if unit in {"minute", "hour", "day"}:
            half = _bucket_seconds_for_fixed_freq(n, unit) / 2.0
            return (
                f"CASE "
                f"WHEN {quoted_source} IS NULL THEN NULL "
                f"WHEN {offset_seconds} >= {half} THEN {ceil_next} "
                f"ELSE {floor_sql} "
                f"END"
            )
        bucket_seconds = f"EPOCH({ceil_next} - {floor_sql})"
        return (
            f"CASE "
            f"WHEN {quoted_source} IS NULL THEN NULL "
            f"WHEN {offset_seconds} * 2 >= {bucket_seconds} THEN {ceil_next} "
            f"ELSE {floor_sql} "
            f"END"
        )
