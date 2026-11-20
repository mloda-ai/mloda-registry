"""Pure Python reference for time bucketization: runs the group's loop with a Python compute.

Math runs on the wall-clock value of the column's own zone; tzinfo is re-attached at the end.
Only ms and us timestamps with no zone, UTC or a fixed offset are in scope; month/year round on non-UTC zones is unjudged.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

from mloda.provider import FeatureSet

from mloda.testing.feature_groups.data_operations.row_preserving.bucket_helpers import bucket_index

if TYPE_CHECKING:
    import pyarrow as pa

TIME_BUCKET_OP_NAMES = frozenset({"floor", "ceil", "round"})
_EPOCH = datetime(1970, 1, 1)
_MICRO = timedelta(microseconds=1)
_UNIT_MICROS = {"minute": 60 * 10**6, "hour": 3600 * 10**6, "day": 86400 * 10**6}
_FIXED_OFFSET = re.compile(r"^[+-]\d{2}:\d{2}$")


def _assert_timestamp(table: pa.Table, source_col: str) -> None:
    import pyarrow as pa

    arrow_type = table.column(source_col).type
    if not pa.types.is_timestamp(arrow_type):
        raise ValueError(f"Source column {source_col!r} must be a timestamp for time bucketization; got {arrow_type}.")
    if arrow_type.unit not in ("ms", "us"):
        raise ValueError(f"Source column {source_col!r} has unit {arrow_type.unit!r}; the reference takes ms or us.")
    tz = arrow_type.tz
    if tz is not None and tz != "UTC" and not _FIXED_OFFSET.match(tz):
        raise ValueError(
            f"Source column {source_col!r} has timezone {tz!r}; the reference takes UTC or a fixed offset."
        )


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


def _bucket(value: datetime, op: str, n: int, unit: str) -> datetime:
    wall = value.replace(tzinfo=None)
    floor, upper = _bounds(wall, n, unit)
    if op == "floor":
        result = floor
    elif op == "ceil":
        result = wall if floor == wall and unit in _UNIT_MICROS else upper
    else:
        result = upper if 2 * (wall - floor) >= upper - floor else floor
    return result.replace(tzinfo=value.tzinfo)


def _compute(table: pa.Table, feature_name: str, source_col: str, op: str, n: int, unit: str) -> pa.Table:
    import pyarrow as pa

    if op not in TIME_BUCKET_OP_NAMES:
        raise ValueError(f"Unsupported bucket op {op!r}. Supported: {sorted(TIME_BUCKET_OP_NAMES)}")
    column = table.column(source_col)
    tz = column.type.tz
    if op == "round" and unit in ("month", "year") and tz is not None and tz != "UTC":
        raise ValueError(
            f"round with unit {unit!r} on timezone {tz!r}: backends disagree there, so the reference leaves it unjudged."
        )
    values = [None if v is None else _bucket(v, op, n, unit) for v in column.to_pylist()]
    return table.append_column(feature_name, pa.array(values, type=column.type))


def reference_time_bucketization(group: Any, table: pa.Table, features: FeatureSet) -> pa.Table:
    """The table plus one pure Python column per bucketization feature."""
    return group.bucketize(table, features, assert_timestamp=_assert_timestamp, compute_bucket=_compute)
