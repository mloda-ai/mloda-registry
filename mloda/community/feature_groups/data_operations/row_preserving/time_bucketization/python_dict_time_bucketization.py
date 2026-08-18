"""PythonDict implementation of time bucketization.

Bucket math runs directly on native ``datetime.datetime`` objects, which
(unlike SQLite's TEXT-stored offsets) carry genuine ``zoneinfo.ZoneInfo``
tzinfo, so DST-aware offsets recompute automatically and non-UTC tz-aware
sources need no special-casing.

Floors: minute/hour/day are all epoch-anchored (multiples of the bucket
duration since 1970-01-01), matching PyArrow's ``floor_temporal`` -- see
``_wall_clock_epoch_seconds`` for why the anchor is computed from *wall-clock*
fields rather than the true UTC instant. Week is ISO-Monday; month/year use a
calendar ``.replace(...)`` floor. ``ceil`` derives from floor plus one
bucket, matching PyArrow's ``ceil_temporal`` convention; ``round`` is
half-up, matching PyArrow's ``round_temporal`` default, with the
floor-to-midpoint distance measured in true elapsed seconds (see
``_elapsed_seconds``) so a DST-shortened/lengthened day is not silently
treated as exactly 24 hours.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
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

_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)

# Fixed-duration units bucketed by an epoch-anchored floor (see _floor_dt).
_SECONDS_PER_UNIT: dict[str, int] = {"minute": 60, "hour": 3600, "day": 86400}


def _attach_tzinfo(naive: datetime, tzinfo: Any) -> datetime:
    """Attach *tzinfo* to a naive *naive* datetime, resolving its DST offset correctly for *naive*'s own date.

    ``zoneinfo.ZoneInfo`` and fixed-offset ``datetime.timezone`` resolve their UTC offset dynamically
    whenever queried, so a plain ``.replace(tzinfo=...)`` is already correct for them. ``pytz``'s
    ``DstTzInfo``, however, pins its offset at construction/``.localize()`` time and does NOT
    recompute it on ``.replace()`` -- so reattaching it via ``.replace(tzinfo=...)`` would silently
    carry over whatever offset happened to be baked into the *original* instance, even onto a
    different calendar date whose real offset differs (e.g. a DST transition boundary). Duck-typing
    on ``localize`` (pytz's public re-localization API) re-resolves the correct offset for pytz
    tzinfo while leaving every other tzinfo type's already-correct ``.replace()`` behavior untouched.
    """
    if tzinfo is None:
        return naive
    localize = getattr(tzinfo, "localize", None)
    if callable(localize):
        localized: datetime = localize(naive)
        return localized
    return naive.replace(tzinfo=tzinfo)


def _wall_clock_epoch_seconds(dt: datetime) -> int:
    """Whole seconds between 1970-01-01T00:00:00 and *dt*'s wall-clock fields.

    Computed by discarding ``dt``'s real ``tzinfo`` and treating its
    calendar/clock fields as if they were already UTC. PyArrow's
    ``floor/ceil/round_temporal`` bucket minute/hour/day by a timestamp's
    *local* wall-clock representation, not its true UTC instant: e.g. 08:37
    local in a UTC+05:30 zone floors a 5-hour bucket to 08:00 local, not to
    the real-instant-equivalent 08:30. Anchoring this wall-clock value to the
    Unix epoch (rather than resetting within the enclosing hour/day, the
    previous bug) makes the floor correct for any ``n``, including values
    that don't evenly divide 60 (minute) or 24 (hour).
    """
    delta = dt.replace(tzinfo=timezone.utc) - _EPOCH_UTC
    return delta.days * 86400 + delta.seconds


def _elapsed_seconds(later: datetime, earlier: datetime) -> float:
    """True elapsed real-world seconds from *earlier* to *later*.

    Uses ``datetime.timestamp()`` rather than ``later - earlier``:
    ``datetime.timestamp()`` always resolves against the UTC epoch (a
    datetime with a *different* tzinfo object), so it correctly reflects any
    DST-driven UTC-offset change. Plain subtraction, by contrast, ignores
    ``tzinfo`` -- and any offset change it represents -- whenever both
    operands share the *same* tzinfo object (e.g. two datetimes both derived
    from one ``zoneinfo.ZoneInfo`` instance via ``_floor_dt``/
    ``_next_boundary``), silently assuming every local day/hour has the same
    real length. A day that crosses a DST transition is not 24 real hours;
    without this fix that error would leak into ``_round_dt``'s midpoint
    comparison.
    """

    def _epoch(dt: datetime) -> float:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc).timestamp()
        return dt.timestamp()

    return _epoch(later) - _epoch(earlier)


def _floor_dt(dt: datetime, n: int, unit: str) -> datetime:
    """Floor ``dt`` to the start of its ``(n, unit)`` bucket, preserving tzinfo.

    All calendar math below is done on *naive* datetimes, with ``dt.tzinfo`` reattached (via
    ``_attach_tzinfo``) only once the final floored wall-clock date is known -- never carried
    through intermediate ``.replace()``/arithmetic steps -- so a DST-crossing floor gets the
    correct offset for its *own* date rather than the offset baked into ``dt``'s original tzinfo
    instance (see ``_attach_tzinfo``).
    """
    tzinfo = dt.tzinfo
    if unit in _SECONDS_PER_UNIT:
        bucket_seconds = n * _SECONDS_PER_UNIT[unit]
        floored_seconds = (_wall_clock_epoch_seconds(dt) // bucket_seconds) * bucket_seconds
        naive = (_EPOCH_UTC + timedelta(seconds=floored_seconds)).replace(tzinfo=None)
        return _attach_tzinfo(naive, tzinfo)
    if unit == "week":
        day_floor_naive = _floor_dt(dt, 1, "day").replace(tzinfo=None)
        naive = day_floor_naive - timedelta(days=day_floor_naive.weekday())
        return _attach_tzinfo(naive, tzinfo)
    if unit == "month":
        naive = dt.replace(tzinfo=None, day=1, hour=0, minute=0, second=0, microsecond=0)
        return _attach_tzinfo(naive, tzinfo)
    if unit == "year":
        naive = dt.replace(tzinfo=None, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        return _attach_tzinfo(naive, tzinfo)
    raise ValueError(f"Unsupported time bucketization unit for PythonDict: {unit!r}")


def _next_boundary(floored: datetime, n: int, unit: str) -> datetime:
    """The start of the bucket immediately after ``floored`` (which must already be floored).

    Like ``_floor_dt``, arithmetic runs on a naive copy of ``floored`` with ``floored.tzinfo``
    reattached only at the end, so a bucket that straddles a DST transition gets the boundary's
    own correct offset instead of inheriting whatever offset ``floored``'s tzinfo instance happens
    to be pinned to.
    """
    tzinfo = floored.tzinfo
    naive = floored.replace(tzinfo=None)
    if unit == "minute":
        return _attach_tzinfo(naive + timedelta(minutes=n), tzinfo)
    if unit == "hour":
        return _attach_tzinfo(naive + timedelta(hours=n), tzinfo)
    if unit == "day":
        return _attach_tzinfo(naive + timedelta(days=n), tzinfo)
    if unit == "week":
        return _attach_tzinfo(naive + timedelta(days=7), tzinfo)
    if unit == "month":
        # floored always has day=1, so no day-of-month clamping is needed.
        year, month = naive.year, naive.month + 1
        if month > 12:
            year, month = year + 1, 1
        return _attach_tzinfo(naive.replace(year=year, month=month), tzinfo)
    if unit == "year":
        return _attach_tzinfo(naive.replace(year=naive.year + 1), tzinfo)
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
    offset = _elapsed_seconds(dt, floored)
    length = _elapsed_seconds(next_boundary, floored)
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
        """Reject ``datetime.date`` (DATE-only, no time component) sources; mirrors the DuckDB DATE-affinity guard."""
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
