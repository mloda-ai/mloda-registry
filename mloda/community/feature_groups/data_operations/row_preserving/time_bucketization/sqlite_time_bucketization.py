"""SQLite implementation of time bucketization.

SQLite has no native timestamp type: timestamps are stored as TEXT in ISO
8601 form (``2023-01-01 00:00:00+00:00`` for tz-aware input). Bucketization
math is expressed with ``strftime`` / ``date`` / ``datetime`` / ``julianday``.

Result-type fidelity
--------------------

The SQL emits tz-aware ISO 8601 strings directly: bucket math runs on the
source-local wall-clock portion of the timestamp (the input with any
``+HH:MM`` / ``-HH:MM`` suffix stripped), then the original tz suffix is
concatenated back onto the result. The output column comes back from
``to_arrow_table()`` as ``pa.string()`` -- there is no Python-side re-parse
into ``pa.timestamp`` and no ``SqliteRelation`` subclass. Null sources
propagate as ``NULL`` (Python ``None``) end-to-end, because SQLite's
``NULL || anything`` is ``NULL``.

Ordering
--------

``calculate_feature`` is invoked many times in a single mloda run, so the
SQL projection assigns ``ROW_NUMBER() OVER (ORDER BY rowid)`` and re-sorts
on it before fetching, mirroring the tag-and-restore pattern in
``sqlite_datetime.py``.
"""

from __future__ import annotations

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


def _zero_pad_two(int_expr: str) -> str:
    """SQL expression that left-pads a non-negative integer expression to two digits.

    SQLite has no ``printf('%02d', ...)`` shorthand here so we synthesize it via
    ``substr('00' || x, -2, 2)``. Extracted because the minute and hour
    ``n > 1`` floor branches both need it.
    """
    return f"substr('00' || {int_expr}, -2, 2)"


# Map a (unit, n) pair to the strftime / date / datetime expression that
# yields the bucket-aligned timestamp string for floor. Calendar-unit n=1
# cases use strftime-style truncation. Fixed-freq (n > 1) cases use bucket
# math on the appropriate field.
def _floor_expr(quoted_source: str, n: int, unit: str) -> str:
    """SQL expression that floors ``quoted_source`` to a ``(n, unit)`` bucket.

    All branches return a ``YYYY-MM-DD HH:MM:SS`` string (no tz suffix);
    null input propagates because ``strftime`` of null is null. ``quoted_source``
    is expected to be a *local-time* expression (tz suffix already stripped by
    the caller).
    """
    if unit == "minute":
        if n == 1:
            return f"strftime('%Y-%m-%d %H:%M:00', {quoted_source})"
        # Floor minutes to multiples of n.
        bucket_minute = f"((cast(strftime('%M', {quoted_source}) as integer) / {n}) * {n})"
        return f"strftime('%Y-%m-%d %H:', {quoted_source}) || {_zero_pad_two(bucket_minute)} || ':00'"
    if unit == "hour":
        if n == 1:
            return f"strftime('%Y-%m-%d %H:00:00', {quoted_source})"
        bucket_hour = f"((cast(strftime('%H', {quoted_source}) as integer) / {n}) * {n})"
        return f"strftime('%Y-%m-%d ', {quoted_source}) || {_zero_pad_two(bucket_hour)} || ':00:00'"
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


def _local_src_expr(quoted_source: str) -> str:
    """SQL expression that strips a trailing ``+HH:MM`` / ``-HH:MM`` tz suffix.

    SQLite has no inline ``LET``, so the caller substitutes this expression
    everywhere a wall-clock view of the source is needed.
    """
    return (
        f"(CASE WHEN substr({quoted_source}, -6, 1) IN ('+', '-') "
        f"THEN substr({quoted_source}, 1, length({quoted_source}) - 6) "
        f"ELSE {quoted_source} END)"
    )


def _tz_suffix_expr(quoted_source: str) -> str:
    """SQL expression that yields the trailing ``+HH:MM`` / ``-HH:MM`` suffix or ``''``."""
    return (
        f"(CASE WHEN substr({quoted_source}, -6, 1) IN ('+', '-') "
        f"THEN substr({quoted_source}, -6) "
        f"ELSE '' END)"
    )


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
            raise ValueError(
                f"Source column {source_col!r} is not present in the SQLite table; "
                f"available: {list(affinity_by_column)}."
            )
        if affinity not in _SQLITE_TIMESTAMP_AFFINITIES:
            cls._raise_non_timestamp_source(source_col, f"SQLite affinity {affinity!r}")

        # Probe a sample non-null value via ``julianday`` to weed out
        # TEXT-affinity columns that hold non-timestamp content. ``LIMIT 1``
        # is intentional: scanning every row to validate would dominate the
        # cost of bucketization itself. A column whose first non-null parses
        # but whose later rows don't will fail at compute time with the
        # underlying SQL error -- accepted trade-off.
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
        local_src = _local_src_expr(quoted_source)
        tz_suffix = _tz_suffix_expr(quoted_source)
        floor_expr = _floor_expr(local_src, n, unit)

        if op == "floor":
            # Floor has no internal null guard, so wrap it here. Note that
            # ``NULL || tz_suffix`` is NULL in SQLite, so this guard primarily
            # documents intent for callers reading the SQL.
            bucket_expr = f"CASE WHEN {quoted_source} IS NULL THEN NULL ELSE {floor_expr} || {tz_suffix} END"
        elif op == "ceil":
            ceil_expr = cls._ceil_expression(local_src, n, unit, floor_expr)
            # ceil_expr already returns NULL for null input; NULL || tz_suffix is NULL.
            bucket_expr = f"({ceil_expr}) || {tz_suffix}"
        else:  # round
            round_expr = cls._round_expression(local_src, n, unit, floor_expr)
            bucket_expr = f"({round_expr}) || {tz_suffix}"

        # Compute the result via a tag-and-restore SQL projection. ROW_NUMBER
        # over rowid preserves input order; the outer SELECT sorts on it.
        rn = "__mloda_rn__"
        qrn = quote_ident(rn)
        quoted_feature = quote_ident(feature_name)
        quoted_table = quote_ident(data.table_name)
        existing_cols = ", ".join(quote_ident(c) for c in data.columns)

        sql = (
            f"SELECT {existing_cols}, {bucket_expr} AS {quoted_feature} "  # nosec
            f"FROM (SELECT *, ROW_NUMBER() OVER (ORDER BY rowid) AS {qrn} FROM {quoted_table}) "
            f"ORDER BY {qrn}"
        )
        cursor = data.connection.execute(sql)
        rows = cursor.fetchall()
        result_values = [row[-1] for row in rows]

        return data.append_column(feature_name, result_values)

    @classmethod
    def _ceil_expression(cls, quoted_source: str, n: int, unit: str, floor_expr: str) -> str:
        """SQL expression for ceil bucketization.

        ``quoted_source`` here is the *local-time* expression (tz suffix stripped).

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
            # against the floored value. ``quoted_source`` is already local-time, so this
            # produces an apples-to-apples comparison with the floor expression.
            f"WHEN datetime({quoted_source}) = {floor_expr} THEN {floor_expr} "
            f"ELSE {ceil_next} "
            f"END"
        )

    @classmethod
    def _round_expression(cls, quoted_source: str, n: int, unit: str, floor_expr: str) -> str:
        """SQL expression for round-half-up bucketization.

        ``quoted_source`` here is the *local-time* expression (tz suffix stripped).

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
