"""Hand-computed pinning rows of the datetime reference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from types import MappingProxyType

from mloda.user import DataType

from mloda.testing.feature_groups.data_operations.pinning import PinningRow, PinningTable

DATETIME_RULES: Mapping[str, str] = MappingProxyType(
    {
        "none": "None in gives None out, for every operation.",
        "leap_day": "Fields are the calendar fields of the value: 2024-02-29 is a Thursday in quarter 1.",
        "pre_epoch": "Values before 1970 extract like any other, so 1969-12-31 is a Wednesday in quarter 4.",
        "sub_second": "second truncates sub-second parts, it never rounds up.",
        "weekday": "dayofweek runs 0 (Monday) to 6 (Sunday); is_weekend is 1 for Saturday and Sunday, else 0.",
        "quarter": "quarter is (month - 1) // 3 + 1, so Mar 31 is 1, Apr 1 is 2 and Dec 31 is 4.",
        "timezone": "A timezone-aware column gives the fields of its own timezone, not of UTC.",
        "dtype": "Every operation gives int64, is_weekend included.",
        "row_order": "Output rows keep the input order; an empty input gives empty int64 columns.",
    }
)

_I = DataType.INT64
_NAIVE = DataType.TIMESTAMP_MICROS
_AWARE = DataType.TIMESTAMP_MICROS_TZ

_OPS = ("year", "month", "day", "hour", "minute", "second", "dayofweek", "is_weekend", "quarter")
_Fields = tuple[int, int, int, int, int, int, int, int, int]
_Value = tuple[datetime, _Fields] | None

_PLUS_5 = timezone(timedelta(hours=5))

# (timestamp, fields in _OPS order), computed by hand.
_LEAP: _Value = (datetime(2024, 2, 29, 13, 45, 59, 999999), (2024, 2, 29, 13, 45, 59, 3, 0, 1))
_PRE_EPOCH: _Value = (datetime(1969, 12, 31, 23, 59, 59, 999999), (1969, 12, 31, 23, 59, 59, 2, 0, 4))
_SAT: _Value = (datetime(2024, 6, 1, 0, 0, 0), (2024, 6, 1, 0, 0, 0, 5, 1, 2))
_SUN: _Value = (datetime(2024, 6, 2, 12, 0, 0), (2024, 6, 2, 12, 0, 0, 6, 1, 2))
_MON: _Value = (datetime(2024, 6, 3, 8, 30, 0), (2024, 6, 3, 8, 30, 0, 0, 0, 2))
_MAR_31: _Value = (datetime(2024, 3, 31, 23, 59, 59), (2024, 3, 31, 23, 59, 59, 6, 1, 1))
_APR_1: _Value = (datetime(2024, 4, 1, 0, 0, 0), (2024, 4, 1, 0, 0, 0, 0, 0, 2))
_DEC_31: _Value = (datetime(2024, 12, 31, 23, 59, 59), (2024, 12, 31, 23, 59, 59, 1, 0, 4))
_SUB_ROUND: _Value = (datetime(2024, 1, 1, 0, 0, 0, 999999), (2024, 1, 1, 0, 0, 0, 0, 0, 1))
_SUB_HALF: _Value = (datetime(2024, 1, 1, 0, 0, 1, 500000), (2024, 1, 1, 0, 0, 1, 0, 0, 1))
_UTC_LEAP: _Value = (datetime(2024, 2, 29, 13, 45, 59, 999999, tzinfo=timezone.utc), (2024, 2, 29, 13, 45, 59, 3, 0, 1))
_PLUS_5_LATE: _Value = (datetime(2024, 3, 31, 22, 30, 15, tzinfo=_PLUS_5), (2024, 3, 31, 22, 30, 15, 6, 1, 1))
_PLUS_5_NEXT_DAY: _Value = (datetime(2024, 4, 1, 2, 0, 0, tzinfo=_PLUS_5), (2024, 4, 1, 2, 0, 0, 0, 0, 2))


def _rows(
    case: str, data_type: DataType, values: Sequence[_Value], ops: Sequence[str] = _OPS
) -> tuple[PinningRow, ...]:
    rows = []
    for op in ops:
        feature = f"ts__{op}"
        index = _OPS.index(op)
        rows.append(
            PinningRow(
                case=case,
                feature=feature,
                inputs={"ts": (data_type, tuple(None if v is None else v[0] for v in values))},
                expected={feature: (_I, tuple(None if v is None else v[1][index] for v in values))},
            )
        )
    return tuple(rows)


DATETIME_PINNING = PinningTable(
    operation="datetime",
    rules=DATETIME_RULES,
    rows=(
        *_rows("none", _NAIVE, [_LEAP, None]),
        *_rows("none", _NAIVE, [None, None]),
        *_rows("none", _AWARE, [None, _UTC_LEAP]),
        *_rows("leap_day", _NAIVE, [_LEAP]),
        *_rows("pre_epoch", _NAIVE, [_PRE_EPOCH]),
        *_rows("sub_second", _NAIVE, [_SUB_ROUND, _SUB_HALF, _LEAP], ops=("second",)),
        *_rows("weekday", _NAIVE, [_SAT, _SUN, _MON], ops=("dayofweek", "is_weekend", "day")),
        *_rows("quarter", _NAIVE, [_MAR_31, _APR_1, _DEC_31], ops=("quarter", "month", "dayofweek")),
        *_rows("timezone", _AWARE, [_UTC_LEAP]),
        *_rows("timezone", _AWARE, [_PLUS_5_LATE, _PLUS_5_NEXT_DAY, None]),
        *_rows("dtype", _NAIVE, [_SUN]),
        *_rows("row_order", _NAIVE, [_MON, _SAT, _SUN], ops=("day", "dayofweek")),
        *_rows("row_order", _NAIVE, [], ops=_OPS),
        *_rows("row_order", _AWARE, [], ops=("year", "is_weekend")),
    ),
)
