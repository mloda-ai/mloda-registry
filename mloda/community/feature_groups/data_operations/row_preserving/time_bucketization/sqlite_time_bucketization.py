"""SQLite implementation of time bucketization.

SQLite has no native timestamp type: timestamps are stored as TEXT in ISO
8601 form (``2023-01-01 00:00:00+00:00`` for tz-aware input). Bucketization
math is expressed with ``strftime`` / ``date`` / ``datetime`` / ``julianday``.

Timezone restriction
--------------------

Only **UTC** and **naive** (no tz) sources are accepted. Non-UTC tz-aware
sources are rejected up-front with a clear ``ValueError``: once a tz-aware
Arrow timestamp is stored as TEXT the IANA zone name is lost and only the
numeric offset survives, which is not enough to do DST-correct bucket math.
For example, ``Europe/Berlin 2023-03-31 14:00+02:00`` floors to
``2023-03-01 00:00+01:00`` in PyArrow (the March 1 offset, pre-DST), but a
wall-clock-and-reattach strategy on the stored TEXT would produce
``+02:00`` for the result. Rather than silently mis-compute, SQLite refuses
the input and the caller is expected to convert to UTC before storing.

Result-type fidelity
--------------------

The SQL emits ISO 8601 strings directly: bucket math runs on the wall-clock
portion of the timestamp (UTC ``+00:00`` suffix stripped if present), then
the original suffix is concatenated back onto the result. The output column
comes back from ``to_arrow_table()`` as ``pa.string()`` -- there is no
Python-side re-parse into ``pa.timestamp`` and no ``SqliteRelation``
subclass. Null sources propagate as ``NULL`` (Python ``None``) end-to-end,
because SQLite's ``NULL || anything`` is ``NULL``.

Bucket math
-----------

Minute / hour / day buckets run on integer epoch microseconds of the wall
clock (``m``), so they are multiples of ``n`` units counted from
1970-01-01 00:00 and floor goes down before 1970. With width ``w``:
``floor = m - ((m % w) + w) % w``; ceil keeps an aligned ``m`` and otherwise
adds ``w``; round adds ``w`` when ``2 * (m - floor) >= w`` (ties up). The
seconds come from the 19-char ``YYYY-MM-DD HH:MM:SS`` slice and the fraction
is parsed separately, because older SQLite rounds fractional seconds to
milliseconds. Week / month / year floors use the same 19-char slice and the
calendar ceil always advances one bucket.

Ordering
--------

The SQL projection is a plain scan without ``ORDER BY``, the order
``append_column`` aligns values by.
"""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.sql.sql_utils import quote_ident
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_framework import SqliteFramework
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.base import (
    TIME_BUCKETIZATION_OPS,
    TimeBucketizationFeatureGroup,
)
from mloda.community.feature_groups.data_operations.sqlite_helpers import (
    column_affinities,
    first_storage_class_outside,
    offset_suffix_expr,
    wall_clock_expr,
)

# Affinity strings (from PRAGMA table_info) that imply a timestamp source.
# ``SqliteRelation.from_arrow`` maps a tz-aware ``pa.timestamp(..., tz="UTC")``
# arrow type to SQLite ``TEXT`` (see ``_arrow_type_to_sqlite``), and the
# stored content is the ISO-8601 representation. We accept TEXT as a
# candidate timestamp affinity and let the SQL functions surface any
# parse-time error from non-timestamp text.
_SQLITE_TIMESTAMP_AFFINITIES: frozenset[str] = frozenset({"TEXT", "DATETIME", "TIMESTAMP", "DATE"})
_TIMESTAMP_STORAGE_CLASSES: frozenset[str] = frozenset({"text"})


_MICROS = 1_000_000
_FIXED_UNIT_MICROS: dict[str, int] = {"minute": 60 * _MICROS, "hour": 3600 * _MICROS, "day": 86400 * _MICROS}


def _wall_micros_expr(wall: str) -> str:
    """SQL expression for the wall clock as integer epoch microseconds (NULL for NULL)."""
    # Always the 19-char slice: older SQLite rounds fractional seconds to ms.
    secs = f"cast(strftime('%s', substr({wall}, 1, 19)) as integer)"
    frac = (
        f"CASE WHEN substr({wall}, 20, 1) = '.' "
        f"THEN cast(substr(substr({wall}, 21) || '000000', 1, 6) as integer) ELSE 0 END"
    )
    return f"({secs} * {_MICROS} + {frac})"


def _micros_to_text(micros: str) -> str:
    """``YYYY-MM-DD HH:MM:SS`` text for boundary microseconds (whole minutes, so exact)."""
    return f"datetime(({micros}) / {_MICROS}, 'unixepoch')"


def _fixed_expr(wall: str, op: str, n: int, unit: str) -> str:
    """Floor / ceil / round for minute, hour and day on integer epoch microseconds."""
    w = n * _FIXED_UNIT_MICROS[unit]
    m = _wall_micros_expr(wall)
    floor = f"({m} - (({m} % {w}) + {w}) % {w})"
    if op == "floor":
        result = floor
    elif op == "ceil":
        result = f"(CASE WHEN {m} = {floor} THEN {floor} ELSE {floor} + {w} END)"
    else:
        result = f"(CASE WHEN 2 * ({m} - {floor}) >= {w} THEN {floor} + {w} ELSE {floor} END)"
    return _micros_to_text(result)


def _calendar_floor_expr(wall: str, unit: str) -> str:
    """Floor of a week / month / year bucket as ``YYYY-MM-DD HH:MM:SS`` text."""
    src = f"substr({wall}, 1, 19)"
    if unit == "week":
        # ISO Monday: %w returns 0=Sunday..6=Saturday, so days_since_monday = (w + 6) % 7.
        offset = f"(cast(strftime('%w', {src}) as integer) + 6) % 7"
        return f"strftime('%Y-%m-%d 00:00:00', date({src}, '-' || ({offset}) || ' days'))"
    if unit == "month":
        return f"strftime('%Y-%m-01 00:00:00', {src})"
    if unit == "year":
        return f"strftime('%Y-01-01 00:00:00', {src})"
    raise ValueError(f"Unsupported time bucketization unit for SQLite: {unit!r}")


def _calendar_modifier(unit: str) -> str:
    """SQLite date modifier for one calendar bucket."""
    return {"week": "'+7 days'", "month": "'+1 month'", "year": "'+1 year'"}[unit]


def _calendar_expr(wall: str, op: str, unit: str) -> str:
    """Floor / ceil / round for week, month and year."""
    floor = _calendar_floor_expr(wall, unit)
    if op == "floor":
        return floor
    upper = f"datetime({floor}, {_calendar_modifier(unit)})"
    if op == "ceil":
        # Always advances, even on aligned input.
        return f"CASE WHEN {wall} IS NULL THEN NULL ELSE {upper} END"
    offset_seconds = f"(julianday({wall}) - julianday({floor})) * 86400.0"
    bucket_seconds = f"(julianday({upper}) - julianday({floor})) * 86400.0"
    return (
        f"CASE WHEN {wall} IS NULL THEN NULL "
        f"WHEN {offset_seconds} * 2 >= {bucket_seconds} THEN {upper} ELSE {floor} END"
    )


class SqliteTimeBucketization(Implementation[TimeBucketizationFeatureGroup]):
    """SQLite backend for time bucketization."""

    feature_group = TimeBucketizationFeatureGroup
    compute_framework = SqliteFramework

    @classmethod
    def calculate(
        cls,
        group: type[TimeBucketizationFeatureGroup],
        data: Any,
        features: FeatureSet,
        framework: ComputeFramework | None,
    ) -> Any:
        return group.bucketize(
            data, features, assert_timestamp=cls._assert_source_column_is_timestamp, compute_bucket=cls._compute_bucket
        )

    @classmethod
    def _assert_source_column_is_timestamp(cls, data: SqliteRelation, source_col: str) -> None:
        """Reject non-timestamp and non-UTC tz-aware source columns.

        Three checks, all required for correct SQLite bucketization:

        1. Column must exist with a timestamp-like affinity (TEXT included,
           since ``SqliteRelation.from_arrow`` maps Arrow timestamps to TEXT).
           A computed column (empty declared type) must hold only text values.
        2. Every non-null value must parse via ``julianday``. SQLite's date
           functions return NULL for unparsable input rather than raising,
           so a LIMIT 1 probe would let later malformed rows slip past and
           silently emit NULL at compute time.
        3. No row may carry a non-UTC offset suffix. SQLite's TEXT storage
           keeps the numeric ``+HH:MM`` but loses the IANA zone, which is
           insufficient for DST-correct bucketization. UTC (``+00:00``)
           and naive (no suffix) are accepted.
        """
        affinity = column_affinities(data)[source_col]
        descriptor = f"SQLite affinity {affinity!r}"
        if affinity == "":
            # Computed column (view expression): no declared type, so probe storage classes.
            descriptor = "SQLite storage class 'text'"
            offender = first_storage_class_outside(data, source_col, _TIMESTAMP_STORAGE_CLASSES)
            if offender is not None:
                TimeBucketizationFeatureGroup._raise_non_timestamp_source(
                    source_col, f"SQLite storage class {offender!r}"
                )
        elif affinity not in _SQLITE_TIMESTAMP_AFFINITIES:
            TimeBucketizationFeatureGroup._raise_non_timestamp_source(source_col, descriptor)

        # Probe every non-null value via ``julianday`` to weed out
        # TEXT-affinity columns that hold non-timestamp content. A LIMIT 1
        # probe is unsafe here: SQLite's date functions return NULL for
        # unparsable input (rather than raising), so a column whose first
        # row parses but whose later rows are malformed would slip past
        # validation and silently emit NULL bucketed values at compute
        # time. We scan the whole column once for any non-null row whose
        # julianday is NULL, and reject if found.
        quoted_source = quote_ident(source_col)
        quoted_table = quote_ident(data.table_name)
        probe_sql = (
            f"SELECT {quoted_source} "  # nosec
            f"FROM {quoted_table} "
            f"WHERE {quoted_source} IS NOT NULL AND julianday({quoted_source}) IS NULL "
            f"LIMIT 1"
        )
        cursor = data.connection.execute(probe_sql)
        row = cursor.fetchone()
        if row is not None:
            (raw_value,) = row
            TimeBucketizationFeatureGroup._raise_non_timestamp_source(
                source_col, f"{descriptor} (value {raw_value!r} does not parse as timestamp)"
            )

        # Reject non-UTC tz-aware sources. SQLite stores tz-aware Arrow
        # timestamps as ISO TEXT with a numeric ``+HH:MM`` / ``-HH:MM``
        # suffix and loses the IANA zone name; without the zone we cannot
        # do DST-correct floor / ceil / round (the result offset depends
        # on the bucketed instant's offset, not the source row's offset).
        # Accept ``+00:00`` (UTC) and naive (no suffix). For anything else
        # we fail loudly rather than silently mis-compute across DST.
        tz_probe_sql = (
            f"SELECT {quoted_source} "  # nosec
            f"FROM {quoted_table} "
            f"WHERE {quoted_source} IS NOT NULL "
            f"AND substr({quoted_source}, -6, 1) IN ('+', '-') "
            f"AND substr({quoted_source}, -6) != '+00:00' "
            f"LIMIT 1"
        )
        cursor = data.connection.execute(tz_probe_sql)
        row = cursor.fetchone()
        if row is not None:
            (raw_value,) = row
            raise ValueError(
                f"Source column {source_col!r} has a non-UTC timezone offset "
                f"(sample value {raw_value!r}); SQLite stores tz-aware timestamps "
                f"as TEXT with only the numeric offset, which is insufficient for "
                f"DST-correct time bucketization. Convert the column to UTC before "
                f"computing bucketization (or use a backend that preserves IANA "
                f"timezones, e.g. PyArrow, Pandas, Polars, DuckDB)."
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
        if op not in TIME_BUCKETIZATION_OPS:
            raise ValueError(f"Unsupported bucket op {op!r} for SQLite; supported: {sorted(TIME_BUCKETIZATION_OPS)}.")

        quoted_source = quote_ident(source_col)
        wall = wall_clock_expr(quoted_source)
        tz_suffix = offset_suffix_expr(quoted_source)
        if unit in _FIXED_UNIT_MICROS:
            bucket = _fixed_expr(wall, op, n, unit)
        else:
            bucket = _calendar_expr(wall, op, unit)
        # NULL || suffix is NULL, so null sources propagate.
        bucket_expr = f"({bucket}) || {tz_suffix}"

        # Positional append: this SELECT must stay free of ORDER BY, JOIN, GROUP BY and DISTINCT.
        quoted_feature = quote_ident(feature_name)
        quoted_table = quote_ident(data.table_name)

        sql = (
            f"SELECT {bucket_expr} AS {quoted_feature} "  # nosec
            f"FROM {quoted_table}"
        )
        cursor = data.connection.execute(sql)
        result_values = [row[0] for row in cursor.fetchall()]

        return data.append_column(feature_name, result_values)
