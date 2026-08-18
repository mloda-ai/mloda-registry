"""PythonDict implementation of time bucketization.

Pure-Python, dependency-free implementation targeting FULL support. All
bucket math runs on native ``datetime.datetime`` objects (as produced by
``pa.Table.to_pylist()`` on the production conversion path), no engine
round-trip required.

Timezone support
-----------------

Unlike SQLite (which stores tz-aware timestamps as TEXT and loses the IANA
zone, forcing a rejection guard for non-UTC offsets), PythonDict's
conversion path yields real Python ``datetime`` objects carrying genuine
``zoneinfo.ZoneInfo`` tzinfo. Bucket math here uses ``.replace(...)`` and
wall-clock ``timedelta`` arithmetic, both of which keep the original tzinfo
object; a subsequent ``.utcoffset()`` call recomputes the correct DST-aware
offset automatically. So non-UTC tz-aware sources are fully supported with
no special-casing.

Source-type guard
------------------

``pa.date32()`` columns round-trip to plain ``datetime.date`` objects (no
``.hour`` / ``.minute`` / ``.second``), so they are rejected up-front with a
clear ``ValueError``, mirroring the DuckDB precedent
(``duckdb_time_bucketization``'s DATE-affinity exclusion).

Bucket math
-----------

- **minute / hour**: bucket floor is computed within the enclosing hour /
  day (``value // n * n``), matching SQLite's / DuckDB's ``n=1`` behaviour
  and agreeing with PyArrow's epoch anchor for any ``n`` that divides 60 /
  24 evenly (the only multiples exercised by the shared test suite).
- **day**: bucket floor is epoch-anchored (multiples of ``n`` days since
  1970-01-01), matching PyArrow's ``floor_temporal`` anchor exactly (see
  ``duckdb_helpers.floor_expr`` for the cross-backend precedent).
- **week**: ISO-Monday floor (day-floor, then subtract ``weekday()`` days).
- **month / year**: calendar floor via ``.replace(...)``, which is what
  makes the DST-crossing month-floor case correct (see module docstring
  above).

``ceil`` derives from ``floor`` plus one bucket: idempotent on aligned
input for fixed-freq units (minute / hour / day), always advances for
calendar units (week / month / year), matching PyArrow's
``ceil_temporal(ceil_is_strictly_greater=False)`` convention. ``round`` is
half-up: the midpoint between floor and the next boundary rounds toward
the next boundary, matching PyArrow's ``round_temporal`` default.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.base import (
    TIME_BUCKETIZATION_OPS,
    TimeBucketizationFeatureGroup,
)

# Calendar units whose ``ceil`` always advances by one bucket even on aligned
# input (matches PyArrow's ``ceil_temporal`` quirk for ``week`` / ``month`` /
# ``year``; fixed-freq units are idempotent on aligned input).
_CALENDAR_CEIL_ALWAYS_ADVANCES: frozenset[str] = frozenset({"week", "month", "year"})

_EPOCH_DATE = date(1970, 1, 1)


def _floor_dt(dt: datetime, n: int, unit: str) -> datetime:
    """Floor ``dt`` to the start of its ``(n, unit)`` bucket, preserving tzinfo."""
    if unit == "minute":
        bucket = (dt.minute // n) * n
        return dt.replace(minute=bucket, second=0, microsecond=0)
    if unit == "hour":
        bucket = (dt.hour // n) * n
        return dt.replace(hour=bucket, minute=0, second=0, microsecond=0)
    if unit == "day":
        # Epoch-anchored (multiples of n days since 1970-01-01), matching
        # PyArrow's floor_temporal bucket alignment.
        local_date = dt.date()
        days_since_epoch = (local_date - _EPOCH_DATE).days
        bucket_days = (days_since_epoch // n) * n
        floor_date = _EPOCH_DATE + timedelta(days=bucket_days)
        return datetime.combine(floor_date, time.min, tzinfo=dt.tzinfo)
    if unit == "week":
        day_floor = _floor_dt(dt, 1, "day")
        return day_floor - timedelta(days=day_floor.weekday())
    if unit == "month":
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if unit == "year":
        return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    raise ValueError(f"Unsupported time bucketization unit for PythonDict: {unit!r}")


def _next_boundary(floored: datetime, n: int, unit: str) -> datetime:
    """The start of the bucket immediately after ``floored`` (which must already be floored)."""
    if unit == "minute":
        return floored + timedelta(minutes=n)
    if unit == "hour":
        return floored + timedelta(hours=n)
    if unit == "day":
        return floored + timedelta(days=n)
    if unit == "week":
        return floored + timedelta(days=7)
    if unit == "month":
        # floored always has day=1, so no day-of-month clamping is needed.
        year, month = floored.year, floored.month + 1
        if month > 12:
            year, month = year + 1, 1
        return floored.replace(year=year, month=month)
    if unit == "year":
        return floored.replace(year=floored.year + 1)
    raise ValueError(f"Unsupported time bucketization unit for PythonDict: {unit!r}")


def _ceil_dt(dt: datetime, n: int, unit: str) -> datetime:
    floored = _floor_dt(dt, n, unit)
    if unit in _CALENDAR_CEIL_ALWAYS_ADVANCES:
        return _next_boundary(floored, n, unit)
    if floored == dt:
        return floored
    return _next_boundary(floored, n, unit)


def _round_dt(dt: datetime, n: int, unit: str) -> datetime:
    floored = _floor_dt(dt, n, unit)
    next_boundary = _next_boundary(floored, n, unit)
    offset = dt - floored
    length = next_boundary - floored
    if offset * 2 >= length:
        return next_boundary
    return floored


_OP_FUNCS: dict[str, Any] = {
    "floor": _floor_dt,
    "ceil": _ceil_dt,
    "round": _round_dt,
}


class PythonDictTimeBucketization(TimeBucketizationFeatureGroup):
    """PythonDict backend for time bucketization."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def _assert_source_column_is_timestamp(cls, data: dict[str, list[Any]], source_col: str) -> None:
        if source_col not in data:
            raise ValueError(
                f"Source column {source_col!r} is not present in the PythonDict data; available: {list(data)}."
            )
        for value in data[source_col]:
            if value is None:
                continue
            if isinstance(value, datetime):
                continue
            if isinstance(value, date):
                cls._raise_non_timestamp_source(
                    source_col,
                    f"Python {type(value).__name__} (DATE-only value {value!r}, no time component; "
                    "convert to datetime before bucketizing)",
                )
            cls._raise_non_timestamp_source(source_col, f"Python {type(value).__name__} (value {value!r})")

    @classmethod
    def _compute_bucket(
        cls,
        data: dict[str, list[Any]],
        feature_name: str,
        source_col: str,
        op: str,
        n: int,
        unit: str,
    ) -> dict[str, list[Any]]:
        fn = _OP_FUNCS.get(op)
        if fn is None:
            raise ValueError(
                f"Unsupported bucket op {op!r} for PythonDict; supported: {sorted(TIME_BUCKETIZATION_OPS)}."
            )

        data = dict(data)
        col = data[source_col]
        data[feature_name] = [fn(value, n, unit) if value is not None else None for value in col]
        return data
