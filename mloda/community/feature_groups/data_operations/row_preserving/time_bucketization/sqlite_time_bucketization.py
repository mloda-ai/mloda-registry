"""SQLite implementation of time bucketization.

SQLite has no native timestamp type: timestamps are stored as TEXT in ISO
8601 form (``2023-01-01 00:00:00+00:00`` for tz-aware input). Bucketization
math is expressed with ``strftime`` / ``date`` / ``datetime`` / ``julianday``.

Result-type fidelity
--------------------

The shared test contract requires the new column to come back from
``to_arrow_table()`` as a list of ``datetime`` objects, but SqliteRelation's
``to_arrow_table`` infers ``pa.string()`` for TEXT columns. The lightweight
:class:`_TimeBucketizationSqliteResult` subclass overrides ``to_arrow_table``
to re-parse the result columns (and only those columns) into
``pa.timestamp("us", tz="UTC")`` arrays. The original timestamp column is
left as a string, matching the rest of SQLite's read contract.

Ordering
--------

``calculate_feature`` is invoked many times in a single mloda run, so the
SQL projection assigns ``ROW_NUMBER() OVER (ORDER BY rowid)`` and re-sorts
on it before fetching, mirroring the tag-and-restore pattern in
``sqlite_datetime.py``.
"""

from __future__ import annotations

import sqlite3

import pyarrow as pa
import pyarrow.compute as pc

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.reserved_columns import assert_no_reserved_columns
from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.base import (
    TIME_BUCKETIZATION_OPS,
    TimeBucketizationFeatureGroup,
)

# Calendar units whose ``ceil`` always advances one bucket even on aligned
# input (PyArrow's behaviour for week / month / year with
# ``ceil_is_strictly_greater=False``). Fixed-freq units are idempotent on
# aligned input.
_CALENDAR_CEIL_ALWAYS_ADVANCES: frozenset[str] = frozenset({"week", "month", "year"})

# Affinity strings (from PRAGMA table_info) that imply a timestamp source.
# ``SqliteRelation.from_arrow`` maps a tz-aware ``pa.timestamp(..., tz="UTC")``
# arrow type to SQLite ``TEXT`` (see ``_arrow_type_to_sqlite``), and the
# stored content is the ISO-8601 representation. We accept TEXT as a
# candidate timestamp affinity and let the SQL functions surface any
# parse-time error from non-timestamp text.
_SQLITE_TIMESTAMP_AFFINITIES: frozenset[str] = frozenset({"TEXT", "DATETIME", "TIMESTAMP", "DATE"})


# Map a (unit, n) pair to the strftime / date / datetime expression that
# yields the bucket-aligned timestamp string for floor. Calendar-unit n=1
# cases use strftime-style truncation. Fixed-freq (n > 1) cases use bucket
# math on the appropriate field.
def _floor_expr(quoted_source: str, n: int, unit: str) -> str:
    """SQL expression that floors ``quoted_source`` to a ``(n, unit)`` bucket.

    All branches return a ``YYYY-MM-DD HH:MM:SS`` string (no tz suffix);
    null input propagates because ``strftime`` of null is null.
    """
    if unit == "minute":
        if n == 1:
            return f"strftime('%Y-%m-%d %H:%M:00', {quoted_source})"
        # Floor minutes to multiples of n.
        bucket_minute = f"((cast(strftime('%M', {quoted_source}) as integer) / {n}) * {n})"
        return f"strftime('%Y-%m-%d %H:', {quoted_source}) || substr('00' || {bucket_minute}, -2, 2) || ':00'"
    if unit == "hour":
        if n == 1:
            return f"strftime('%Y-%m-%d %H:00:00', {quoted_source})"
        bucket_hour = f"((cast(strftime('%H', {quoted_source}) as integer) / {n}) * {n})"
        return f"strftime('%Y-%m-%d ', {quoted_source}) || substr('00' || {bucket_hour}, -2, 2) || ':00:00'"
    if unit == "day":
        if n == 1:
            return f"strftime('%Y-%m-%d 00:00:00', {quoted_source})"
        # Bucket on days since 1970-01-01 (Thursday). Note: n=7 is not a
        # week alias in v1; only ``week`` (calendar Monday) is supported.
        bucket = f"cast((julianday(date({quoted_source})) - julianday('1970-01-01')) as integer) / {n}"
        return f"strftime('%Y-%m-%d 00:00:00', date('1970-01-01', '+' || ({bucket} * {n}) || ' days'))"
    if unit == "week":
        # ISO Monday: %w returns 0=Sunday..6=Saturday, so days_since_monday = (w + 6) % 7.
        offset = f"(cast(strftime('%w', {quoted_source}) as integer) + 6) % 7"
        return f"strftime('%Y-%m-%d 00:00:00', date({quoted_source}, '-' || ({offset}) || ' days'))"
    if unit == "month":
        return f"strftime('%Y-%m-01 00:00:00', {quoted_source})"
    if unit == "year":
        return f"strftime('%Y-01-01 00:00:00', {quoted_source})"
    raise ValueError(f"Unsupported time bucketization unit for SQLite: {unit!r}")


def _interval_modifier(n: int, unit: str) -> str:
    """SQLite date/datetime modifier corresponding to one bucket of ``(n, unit)``."""
    if unit == "minute":
        return f"'+{n} minutes'"
    if unit == "hour":
        return f"'+{n} hours'"
    if unit == "day":
        return f"'+{n} days'"
    if unit == "week":
        return "'+7 days'"
    if unit == "month":
        return "'+1 month'"
    if unit == "year":
        return "'+1 year'"
    raise ValueError(f"Unsupported time bucketization unit for SQLite: {unit!r}")


def _bucket_seconds(n: int, unit: str) -> int:
    """Length of one ``(n, unit)`` bucket in seconds, for fixed-freq units only.

    Used by the ``round`` half-up midpoint comparison. Calendar units are
    handled separately because their length varies per row.
    """
    if unit == "minute":
        return 60 * n
    if unit == "hour":
        return 3600 * n
    if unit == "day":
        return 86400 * n
    raise ValueError(f"Bucket-seconds helper called for non-fixed-freq unit: {unit!r}")


class _TimeBucketizationSqliteResult(SqliteRelation):
    """SqliteRelation subclass that recovers timestamp typing for result columns.

    SQLite stores all timestamps as TEXT. The base
    ``SqliteRelation.to_arrow_table`` therefore returns the result column as
    ``pa.string()``, but the cross-framework time-bucketization tests
    compare against tz-aware ``datetime`` objects. This subclass re-parses
    the listed result columns into ``pa.timestamp("us", tz="UTC")`` so
    downstream consumers see a real timestamp type.
    """

    def __init__(
        self,
        connection: sqlite3.Connection,
        table_name: str,
        timestamp_result_columns: frozenset[str],
        _is_view: bool = False,
    ) -> None:
        super().__init__(connection, table_name, _is_view=_is_view)
        self._timestamp_result_columns: frozenset[str] = timestamp_result_columns

    @property
    def timestamp_result_columns(self) -> frozenset[str]:
        return self._timestamp_result_columns

    def to_arrow_table(self) -> pa.Table:
        table = super().to_arrow_table()
        if not self._timestamp_result_columns:
            return table

        result_table = table
        for col_name in table.column_names:
            if col_name not in self._timestamp_result_columns:
                continue
            # Vectorized TEXT -> tz-aware timestamp via pyarrow.compute.
            # SQLite's strftime emits 'YYYY-MM-DD HH:MM:SS' without a tz
            # suffix; the source was UTC, so we attach UTC after parsing.
            text_array = result_table.column(col_name)
            naive = pc.strptime(text_array, format="%Y-%m-%d %H:%M:%S", unit="us", error_is_null=True)
            ts_array = pc.assume_timezone(naive, "UTC")
            col_index = result_table.column_names.index(col_name)
            result_table = result_table.set_column(col_index, col_name, ts_array)
        return result_table


class SqliteTimeBucketization(TimeBucketizationFeatureGroup):
    """SQLite backend for time bucketization."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {SqliteFramework}

    @classmethod
    def _assert_source_column_is_timestamp(cls, data: SqliteRelation, source_col: str) -> None:
        """Reject non-timestamp source columns.

        SQLite stores both string and timestamp arrow inputs as ``TEXT``,
        so PRAGMA affinity alone is insufficient to distinguish them. We
        additionally probe with ``julianday(value)`` on the first non-null
        value: a parseable timestamp returns a real number, anything else
        returns NULL. An all-null column is accepted (no rows to validate).
        """
        rows = data.connection.execute(f"PRAGMA table_info({quote_ident(data.table_name)})").fetchall()
        affinity_by_column = {row[1]: (row[2] or "").upper() for row in rows}
        affinity = affinity_by_column.get(source_col)
        if affinity is None:
            return
        if affinity not in _SQLITE_TIMESTAMP_AFFINITIES:
            cls._raise_non_timestamp_source(source_col, f"SQLite affinity {affinity!r}")

        # Probe a sample non-null value via ``julianday`` to weed out
        # TEXT-affinity columns that hold non-timestamp content.
        quoted_source = quote_ident(source_col)
        quoted_table = quote_ident(data.table_name)
        probe_sql = (
            f"SELECT {quoted_source}, julianday({quoted_source}) "  # nosec
            f"FROM {quoted_table} "
            f"WHERE {quoted_source} IS NOT NULL LIMIT 1"
        )
        cursor = data.connection.execute(probe_sql)
        row = cursor.fetchone()
        if row is None:
            return
        raw_value, julian = row
        if julian is None:
            cls._raise_non_timestamp_source(
                source_col, f"SQLite affinity {affinity!r} (sample value {raw_value!r} does not parse as timestamp)"
            )

    @classmethod
    def _compute_bucket(
        cls,
        data: SqliteRelation,
        feature_name: str,
        source_col: str,
        op: str,
        n: int,
        unit: str,
    ) -> SqliteRelation:
        assert_no_reserved_columns(data.columns, framework="SQLite", operation="time bucketization")

        if op not in TIME_BUCKETIZATION_OPS:
            raise ValueError(f"Unsupported bucket op {op!r} for SQLite; supported: {sorted(TIME_BUCKETIZATION_OPS)}.")

        quoted_source = quote_ident(source_col)
        floor_expr = _floor_expr(quoted_source, n, unit)

        if op == "floor":
            result_expr = floor_expr
        elif op == "ceil":
            result_expr = cls._ceil_expression(quoted_source, n, unit, floor_expr)
        else:  # round
            result_expr = cls._round_expression(quoted_source, n, unit, floor_expr)

        # Compute the result via a tag-and-restore SQL projection. ROW_NUMBER
        # over rowid preserves input order; the outer SELECT sorts on it.
        rn = "__mloda_rn__"
        qrn = quote_ident(rn)
        quoted_feature = quote_ident(feature_name)
        quoted_table = quote_ident(data.table_name)
        existing_cols = ", ".join(quote_ident(c) for c in data.columns)

        sql = (
            f"SELECT {existing_cols}, {result_expr} AS {quoted_feature} "  # nosec
            f"FROM (SELECT *, ROW_NUMBER() OVER (ORDER BY rowid) AS {qrn} FROM {quoted_table}) "
            f"ORDER BY {qrn}"
        )
        cursor = data.connection.execute(sql)
        rows = cursor.fetchall()
        result_values = [row[-1] for row in rows]

        appended = data.append_column(feature_name, result_values)

        # Carry forward any timestamp result columns from a chained call.
        existing_ts: frozenset[str]
        if isinstance(data, _TimeBucketizationSqliteResult):
            existing_ts = data.timestamp_result_columns
        else:
            existing_ts = frozenset()

        return _TimeBucketizationSqliteResult(
            connection=appended.connection,
            table_name=appended.table_name,
            timestamp_result_columns=existing_ts | {feature_name},
            _is_view=appended._is_view,
        )

    @classmethod
    def _ceil_expression(cls, quoted_source: str, n: int, unit: str, floor_expr: str) -> str:
        """SQL expression for ceil bucketization.

        Matches PyArrow's ``ceil_temporal(..., ceil_is_strictly_greater=False)``,
        which is unit-dependent:

        - **Fixed-freq units** (``minute`` / ``hour`` / ``day``): idempotent
          on aligned input (``ceil(aligned) == aligned``).
        - **Calendar units** (``week`` / ``month`` / ``year``): NOT
          idempotent; aligned input advances one full bucket (e.g.
          ``ceil_1_year(2023-01-01) == 2024-01-01``).
        """
        ceil_next = f"datetime({floor_expr}, {_interval_modifier(n, unit)})"
        if unit in _CALENDAR_CEIL_ALWAYS_ADVANCES:
            return f"CASE WHEN {quoted_source} IS NULL THEN NULL ELSE {ceil_next} END"
        return (
            f"CASE "
            f"WHEN {quoted_source} IS NULL THEN NULL "
            # Compare the input timestamp (after coercion to YYYY-MM-DD HH:MM:SS via datetime())
            # against the floored value. The input may carry a tz suffix that the floor expression
            # strips, so coerce the source through datetime() so the comparison is between
            # equivalent local-time strings.
            f"WHEN datetime({quoted_source}) = {floor_expr} THEN {floor_expr} "
            f"ELSE {ceil_next} "
            f"END"
        )

    @classmethod
    def _round_expression(cls, quoted_source: str, n: int, unit: str, floor_expr: str) -> str:
        """SQL expression for round-half-up bucketization.

        For fixed-freq units (minute / hour / day) the midpoint compares
        the offset from the floor against half the bucket length in seconds.
        For calendar units (week / month / year) the midpoint compares the
        offset against half the bucket length, computed from
        ``ceil_next - floored`` (in seconds via ``julianday``).
        """
        ceil_next = f"datetime({floor_expr}, {_interval_modifier(n, unit)})"

        # Offset of the input from the floored bucket, in seconds.
        offset_seconds = f"(julianday({quoted_source}) - julianday({floor_expr})) * 86400.0"

        if unit in {"minute", "hour", "day"}:
            half = _bucket_seconds(n, unit) / 2.0
            return (
                f"CASE "
                f"WHEN {quoted_source} IS NULL THEN NULL "
                f"WHEN {offset_seconds} >= {half!r} THEN {ceil_next} "
                f"ELSE {floor_expr} "
                f"END"
            )

        # Calendar units (week, month, year): bucket length depends on row,
        # so we compute it from ceil_next - floor in seconds.
        bucket_seconds = f"(julianday({ceil_next}) - julianday({floor_expr})) * 86400.0"
        return (
            f"CASE "
            f"WHEN {quoted_source} IS NULL THEN NULL "
            f"WHEN {offset_seconds} * 2 >= {bucket_seconds} THEN {ceil_next} "
            f"ELSE {floor_expr} "
            f"END"
        )
