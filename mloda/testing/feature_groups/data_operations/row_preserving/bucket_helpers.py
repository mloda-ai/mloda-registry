"""Shared by the pure Python references of binning, time bucketization and resample."""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow as pa

_EPOCH = datetime(1970, 1, 1)
_MICRO = timedelta(microseconds=1)
_UNIT_MICROS = {"minute": 60 * 10**6, "hour": 3600 * 10**6, "day": 86400 * 10**6}
_FIXED_OFFSET = re.compile(r"^[+-]\d{2}:\d{2}$")


def bucket_index(offset: int | float, width: int | float) -> int:
    """Floor of offset / width: exact for ints; floats divide first because ``1.0 // 0.1 == 9.0``."""
    if isinstance(offset, int) and isinstance(width, int):
        return offset // width
    return math.floor(offset / width)


def assert_timestamp(
    table: pa.Table, column: str, operation: str = "time bucketization", *, allow_fixed_offset: bool = True
) -> None:
    """Reject a column that is not a ms or us timestamp with no zone, UTC or a fixed offset."""
    import pyarrow as pa

    arrow_type = table.column(column).type
    if not pa.types.is_timestamp(arrow_type):
        raise ValueError(f"Source column {column!r} must be a timestamp for {operation}; got {arrow_type}.")
    if arrow_type.unit not in ("ms", "us"):
        raise ValueError(f"Source column {column!r} has unit {arrow_type.unit!r}; the reference takes ms or us.")
    tz = arrow_type.tz
    if tz is not None and tz != "UTC" and not allow_fixed_offset:
        raise ValueError(f"Source column {column!r} has timezone {tz!r}; the reference takes no zone or UTC.")
    if tz is not None and tz != "UTC" and not _FIXED_OFFSET.match(tz):
        raise ValueError(f"Source column {column!r} has timezone {tz!r}; the reference takes UTC or a fixed offset.")


def _month_start(year: int, month: int) -> datetime:
    return datetime(year + (month - 1) // 12, (month - 1) % 12 + 1, 1)


def _bounds(wall: datetime, n: int, unit: str) -> tuple[datetime, datetime]:
    """The floor and the next boundary of the bucket holding ``wall``."""
    if unit in _UNIT_MICROS:
        width = n * _UNIT_MICROS[unit]
        micros = (wall - _EPOCH) // _MICRO
        floor = _EPOCH + bucket_index(micros, width) * width * _MICRO
        return floor, floor + width * _MICRO
    if unit == "week":
        floor = wall.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=wall.weekday())
        return floor, floor + timedelta(days=7)
    if unit == "month":
        return _month_start(wall.year, wall.month), _month_start(wall.year, wall.month + 1)
    return datetime(wall.year, 1, 1), datetime(wall.year + 1, 1, 1)


def bucket_timestamp(value: datetime, op: str, n: int, unit: str) -> datetime:
    """Floor, ceil or round ``value`` on its wall clock to ``n`` ``unit`` buckets; tzinfo is re-attached."""
    wall = value.replace(tzinfo=None)
    floor, upper = _bounds(wall, n, unit)
    if op == "floor":
        result = floor
    elif op == "ceil":
        result = wall if floor == wall and unit in _UNIT_MICROS else upper
    else:
        result = upper if 2 * (wall - floor) >= upper - floor else floor
    return result.replace(tzinfo=value.tzinfo)
