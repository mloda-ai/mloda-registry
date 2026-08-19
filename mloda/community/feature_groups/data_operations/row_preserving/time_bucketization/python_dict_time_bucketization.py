"""PythonDict implementation of time bucketization.

Bucket math runs directly on native ``datetime.datetime`` objects, which
(unlike SQLite's TEXT-stored offsets) carry genuine ``zoneinfo.ZoneInfo``
tzinfo, so DST-aware offsets recompute automatically and non-UTC tz-aware
sources need no special-casing.

Floors: minute/hour/day are all epoch-anchored (multiples of the bucket
duration since 1970-01-01), matching PyArrow's ``floor_temporal`` -- see
``python_dict_helpers.wall_clock_epoch_seconds`` for why the anchor is
computed from *wall-clock* fields rather than the true UTC instant. Week is
ISO-Monday; month/year use a calendar ``.replace(...)`` floor. ``ceil``
derives from floor plus one bucket, matching PyArrow's ``ceil_temporal``
convention; ``round`` is half-up, matching PyArrow's ``round_temporal``
default, with the floor-to-midpoint distance measured in true elapsed
seconds (see ``_elapsed_seconds``) so a DST-shortened/lengthened day is not
silently treated as exactly 24 hours.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from mloda.community.feature_groups.data_operations.python_dict_helpers import (
    SECONDS_PER_UNIT,
    attach_tzinfo as _attach_tzinfo,
    floor_fixed_duration,
)
from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.base import (
    TIME_BUCKETIZATION_OPS,
    TimeBucketizationFeatureGroup,
)

# Calendar units whose ``ceil`` always advances by one bucket even on aligned
# input (matches PyArrow's ``ceil_temporal`` quirk for ``week`` / ``month`` /
# ``year``; fixed-freq units are idempotent on aligned input).
_CALENDAR_CEIL_ALWAYS_ADVANCES: frozenset[str] = frozenset({"week", "month", "year"})


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
    if unit in SECONDS_PER_UNIT:
        return floor_fixed_duration(dt, n, unit)
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
