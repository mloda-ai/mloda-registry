"""Hand-computed pinning rows of the time bucketization reference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone, tzinfo
from types import MappingProxyType

from mloda.user import DataType

from mloda.testing.feature_groups.data_operations.pinning import PinningRow, PinningTable

TIME_BUCKETIZATION_RULES: Mapping[str, str] = MappingProxyType(
    {
        "none": "None in gives None out, for every operation.",
        "empty": "An empty input gives an empty column of the input type.",
        "wall_clock": "All math runs on the local wall-clock value of the column's own zone, never on UTC.",
        "unit_kept": "The output keeps the input Arrow type: the unit (ms or us) and the timezone.",
        "sub_second": "Microseconds count: a value one microsecond past a boundary is not aligned.",
        "floor": "floor gives the start of the enclosing bucket of minute, hour or day.",
        "floor_week": "week floors to Monday 00:00, so Sunday 2023-01-01 floors to Monday 2022-12-26.",
        "floor_calendar": "month and year floor to the first day of the month or year.",
        "epoch_multiple": (
            "minute, hour and day with n are multiples of n units counted from 1970-01-01 00:00 wall clock."
        ),
        "pre_epoch": "Before 1970 the buckets keep counting backwards, so floor goes down, never toward zero.",
        "ceil_next": "ceil of a non-aligned value gives the next bucket boundary.",
        "ceil_aligned": "ceil of an aligned minute, hour or day value stays.",
        "ceil_calendar": "ceil of an aligned week, month or year value advances to the next bucket.",
        "round_tie": "round gives the next boundary when 2 * (value - floor) >= (next - floor), so ties go up.",
        "round_length": (
            "round compares elapsed time, so bucket length matters: Jan 31 days, Feb 28 or 29, year 365 or 366."
        ),
        "leap_day": "2024-02-29 exists: ceil of the day gives 2024-03-01 and the month has 29 days.",
        "row_order": "Output rows keep the input order.",
    }
)

_NAIVE = DataType.TIMESTAMP_MICROS
_MILLIS = DataType.TIMESTAMP_MILLIS
_AWARE = DataType.TIMESTAMP_MICROS_TZ

_UTC = timezone.utc
_PLUS_5 = timezone(timedelta(hours=5))
_MINUS_5 = timezone(timedelta(hours=-5))
_MINUS_3_30 = timezone(timedelta(hours=-3, minutes=-30))

_Pair = tuple[datetime | None, datetime | None]


def _at(
    zone: tzinfo | None,
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    micro: int = 0,
) -> datetime:
    return datetime(year, month, day, hour, minute, second, micro, tzinfo=zone)


def _n(*fields: int) -> datetime:
    return _at(None, *fields)


def _u(*fields: int) -> datetime:
    return _at(_UTC, *fields)


def _p(*fields: int) -> datetime:
    return _at(_PLUS_5, *fields)


def _f(*fields: int) -> datetime:
    return _at(_MINUS_5, *fields)


def _m(*fields: int) -> datetime:
    return _at(_MINUS_3_30, *fields)


def _row(case: str, token: str, pairs: Sequence[_Pair], data_type: DataType = _NAIVE) -> PinningRow:
    feature = f"ts__{token}"
    return PinningRow(
        case=case,
        feature=feature,
        inputs={"ts": (data_type, tuple(source for source, _ in pairs))},
        expected={feature: (data_type, tuple(result for _, result in pairs))},
    )


TIME_BUCKETIZATION_PINNING = PinningTable(
    operation="time_bucketization",
    rules=TIME_BUCKETIZATION_RULES,
    rows=(
        _row("none", "floor_1_day", [(_n(2024, 3, 15, 10, 20, 45), _n(2024, 3, 15)), (None, None)]),
        _row("none", "ceil_1_hour", [(None, None), (_n(2024, 3, 15, 10, 20), _n(2024, 3, 15, 11))]),
        _row("none", "round_1_month", [(_n(2024, 3, 15, 10), _n(2024, 3, 1)), (None, None)]),
        _row("none", "floor_1_day", [(None, None), (None, None)]),
        _row("none", "floor_1_day", [(None, None), (_u(2024, 3, 15, 10, 20, 45), _u(2024, 3, 15))], _AWARE),
        _row("empty", "floor_1_day", []),
        _row("empty", "ceil_1_week", []),
        _row("empty", "round_1_year", []),
        _row("empty", "floor_1_day", [], _MILLIS),
        # wall clock of the column's own zone
        _row("wall_clock", "floor_1_day", [(_u(2024, 3, 15, 10, 20, 45), _u(2024, 3, 15))], _AWARE),
        _row(
            "wall_clock",
            "floor_1_day",
            [(_p(2024, 1, 1, 3), _p(2024, 1, 1)), (_p(2024, 1, 1, 23), _p(2024, 1, 1))],
            _AWARE,
        ),
        _row("wall_clock", "ceil_1_hour", [(_p(2024, 1, 1, 3, 20), _p(2024, 1, 1, 4))], _AWARE),
        _row(
            "wall_clock",
            "round_1_day",
            [(_p(2024, 1, 1, 12), _p(2024, 1, 2)), (_p(2024, 1, 1, 11), _p(2024, 1, 1))],
            _AWARE,
        ),
        _row("wall_clock", "floor_1_week", [(_p(2024, 3, 17, 2), _p(2024, 3, 11))], _AWARE),
        _row("wall_clock", "ceil_1_month", [(_p(2024, 1, 15), _p(2024, 2, 1))], _AWARE),
        _row("wall_clock", "floor_1_year", [(_p(2024, 6, 15), _p(2024, 1, 1))], _AWARE),
        _row(
            "wall_clock",
            "floor_1_day",
            [(_m(2024, 1, 1, 2), _m(2024, 1, 1)), (_m(2024, 1, 1, 23, 59), _m(2024, 1, 1))],
            _AWARE,
        ),
        _row("wall_clock", "floor_1_hour", [(_m(2024, 1, 1, 2, 45), _m(2024, 1, 1, 2))], _AWARE),
        _row("wall_clock", "ceil_1_hour", [(_m(2024, 1, 1, 2, 45), _m(2024, 1, 1, 3))], _AWARE),
        _row("wall_clock", "round_1_minute", [(_m(2024, 1, 1, 2, 45, 30), _m(2024, 1, 1, 2, 46))], _AWARE),
        # the unit and the zone survive
        _row(
            "unit_kept",
            "floor_1_hour",
            [(_n(2024, 3, 15, 10, 20, 45, 123000), _n(2024, 3, 15, 10)), (None, None)],
            _MILLIS,
        ),
        _row("unit_kept", "ceil_1_minute", [(_n(2024, 3, 15, 10, 20, 45, 123000), _n(2024, 3, 15, 10, 21))], _MILLIS),
        _row("unit_kept", "round_1_day", [(_n(2024, 3, 15, 12, 0, 0, 1000), _n(2024, 3, 16))], _MILLIS),
        _row("unit_kept", "floor_1_day", [(_p(2024, 3, 15, 10), _p(2024, 3, 15))], _AWARE),
        _row("unit_kept", "floor_1_day", [(_n(2024, 3, 15, 10), _n(2024, 3, 15))]),
        # microseconds
        _row("sub_second", "floor_1_minute", [(_n(2024, 3, 15, 10, 20, 45, 500000), _n(2024, 3, 15, 10, 20))]),
        _row(
            "sub_second",
            "ceil_1_minute",
            [
                (_n(2024, 3, 15, 10, 20, 0, 1), _n(2024, 3, 15, 10, 21)),
                (_n(2024, 3, 15, 10, 20), _n(2024, 3, 15, 10, 20)),
            ],
        ),
        _row(
            "sub_second",
            "round_1_minute",
            [
                (_n(2024, 3, 15, 10, 20, 29, 999999), _n(2024, 3, 15, 10, 20)),
                (_n(2024, 3, 15, 10, 20, 30), _n(2024, 3, 15, 10, 21)),
            ],
        ),
        _row("sub_second", "ceil_1_day", [(_n(2024, 3, 15, 0, 0, 0, 1), _n(2024, 3, 16))]),
        # floor
        _row(
            "floor",
            "floor_1_minute",
            [
                (_n(2024, 3, 15, 10, 20, 45), _n(2024, 3, 15, 10, 20)),
                (_n(2024, 3, 15, 10, 21), _n(2024, 3, 15, 10, 21)),
            ],
        ),
        _row(
            "floor",
            "floor_1_hour",
            [(_n(2024, 3, 15, 10, 20, 45), _n(2024, 3, 15, 10)), (_n(2024, 3, 15, 23, 59, 59), _n(2024, 3, 15, 23))],
        ),
        _row(
            "floor", "floor_1_day", [(_n(2024, 3, 15, 10, 20, 45), _n(2024, 3, 15)), (_n(2024, 3, 15), _n(2024, 3, 15))]
        ),
        _row(
            "floor_week",
            "floor_1_week",
            [
                (_n(2024, 3, 15, 10, 20), _n(2024, 3, 11)),
                (_n(2023, 1, 1), _n(2022, 12, 26)),
                (_n(2023, 1, 2, 12), _n(2023, 1, 2)),
                (_n(2024, 3, 11), _n(2024, 3, 11)),
                (_n(2024, 3, 17, 23, 59, 59), _n(2024, 3, 11)),
            ],
        ),
        _row(
            "floor_calendar",
            "floor_1_month",
            [
                (_n(2024, 3, 15, 10, 20), _n(2024, 3, 1)),
                (_n(2024, 12, 31, 23, 59, 59), _n(2024, 12, 1)),
                (_n(2024, 3, 1), _n(2024, 3, 1)),
            ],
        ),
        _row(
            "floor_calendar",
            "floor_1_year",
            [
                (_n(2024, 3, 15), _n(2024, 1, 1)),
                (_n(2023, 12, 31, 23, 59, 59), _n(2023, 1, 1)),
                (_n(2024, 1, 1), _n(2024, 1, 1)),
            ],
        ),
        # multiples of n counted from the epoch
        _row(
            "epoch_multiple",
            "floor_3_day",
            [
                (_n(2024, 1, 1), _n(2023, 12, 31)),
                (_n(2024, 1, 2, 12), _n(2023, 12, 31)),
                (_n(2024, 1, 3, 5), _n(2024, 1, 3)),
            ],
        ),
        _row(
            "epoch_multiple",
            "ceil_3_day",
            [
                (_n(2024, 1, 1), _n(2024, 1, 3)),
                (_n(2024, 1, 3), _n(2024, 1, 3)),
                (_n(2024, 1, 3, 0, 0, 1), _n(2024, 1, 6)),
            ],
        ),
        _row(
            "epoch_multiple",
            "floor_7_minute",
            [
                (_n(2024, 1, 1, 0, 0), _n(2023, 12, 31, 23, 54)),
                (_n(2024, 1, 1, 0, 1), _n(2024, 1, 1, 0, 1)),
                (_n(2024, 1, 1, 0, 7), _n(2024, 1, 1, 0, 1)),
            ],
        ),
        _row(
            "epoch_multiple",
            "round_7_minute",
            [
                (_n(2024, 1, 1, 0, 0), _n(2024, 1, 1, 0, 1)),
                (_n(2024, 1, 1, 0, 4), _n(2024, 1, 1, 0, 1)),
                (_n(2024, 1, 1, 0, 5), _n(2024, 1, 1, 0, 8)),
            ],
        ),
        _row(
            "epoch_multiple",
            "floor_5_hour",
            [
                (_n(2024, 1, 1, 0), _n(2023, 12, 31, 22)),
                (_n(2024, 1, 1, 3, 30), _n(2024, 1, 1, 3)),
                (_n(2024, 1, 1, 4), _n(2024, 1, 1, 3)),
            ],
        ),
        _row(
            "epoch_multiple",
            "ceil_5_hour",
            [(_n(2024, 1, 1, 0), _n(2024, 1, 1, 3)), (_n(2024, 1, 1, 3), _n(2024, 1, 1, 3))],
        ),
        _row("epoch_multiple", "floor_5_minute", [(_n(2024, 6, 15, 14, 37, 23), _n(2024, 6, 15, 14, 35))]),
        _row("epoch_multiple", "floor_3_day", [(_n(1970, 1, 1), _n(1970, 1, 1)), (_n(1970, 1, 3, 23), _n(1970, 1, 1))]),
        # before the epoch
        _row("pre_epoch", "floor_1_day", [(_n(1969, 12, 31, 18), _n(1969, 12, 31))]),
        _row("pre_epoch", "floor_3_day", [(_n(1969, 12, 31), _n(1969, 12, 29))]),
        _row("pre_epoch", "floor_5_minute", [(_n(1969, 12, 31, 23, 59, 30), _n(1969, 12, 31, 23, 55))]),
        _row(
            "pre_epoch",
            "floor_7_minute",
            [
                (_n(1969, 12, 31, 23, 59), _n(1969, 12, 31, 23, 53)),
                (_n(1969, 12, 31, 23, 53), _n(1969, 12, 31, 23, 53)),
            ],
        ),
        _row("pre_epoch", "floor_1_week", [(_n(1969, 12, 31), _n(1969, 12, 29))]),
        _row("pre_epoch", "floor_1_month", [(_n(1969, 12, 15), _n(1969, 12, 1))]),
        _row("pre_epoch", "floor_1_year", [(_n(1969, 6, 15), _n(1969, 1, 1))]),
        _row("pre_epoch", "ceil_1_day", [(_n(1969, 12, 31, 18), _n(1970, 1, 1))]),
        _row("pre_epoch", "round_1_hour", [(_n(1969, 12, 31, 23, 30), _n(1970, 1, 1))]),
        # ceil
        _row(
            "ceil_next",
            "ceil_1_minute",
            [(_n(2024, 3, 15, 10, 20, 45), _n(2024, 3, 15, 10, 21)), (_n(2024, 3, 15, 23, 59, 59), _n(2024, 3, 16))],
        ),
        _row(
            "ceil_next",
            "ceil_1_hour",
            [(_n(2024, 3, 15, 10, 20, 45), _n(2024, 3, 15, 11)), (_n(2024, 3, 15, 23, 30), _n(2024, 3, 16))],
        ),
        _row(
            "ceil_next",
            "ceil_1_day",
            [(_n(2024, 3, 15, 10, 20, 45), _n(2024, 3, 16)), (_n(2024, 12, 31, 23, 59, 59), _n(2025, 1, 1))],
        ),
        _row(
            "ceil_next",
            "ceil_5_minute",
            [
                (_n(2024, 6, 15, 14, 37, 23), _n(2024, 6, 15, 14, 40)),
                (_n(2024, 6, 15, 14, 30, 1), _n(2024, 6, 15, 14, 35)),
            ],
        ),
        _row("ceil_aligned", "ceil_1_minute", [(_n(2024, 3, 15, 10, 20), _n(2024, 3, 15, 10, 20))]),
        _row(
            "ceil_aligned",
            "ceil_1_hour",
            [(_n(2024, 3, 15, 10), _n(2024, 3, 15, 10)), (_n(2024, 3, 15, 10, 0, 1), _n(2024, 3, 15, 11))],
        ),
        _row("ceil_aligned", "ceil_1_day", [(_n(2024, 3, 15), _n(2024, 3, 15))]),
        _row("ceil_aligned", "ceil_5_minute", [(_n(2024, 6, 15, 14, 30), _n(2024, 6, 15, 14, 30))]),
        _row(
            "ceil_calendar",
            "ceil_1_week",
            [
                (_n(2023, 1, 2), _n(2023, 1, 9)),
                (_n(2023, 1, 2, 12), _n(2023, 1, 9)),
                (_n(2023, 1, 1), _n(2023, 1, 2)),
                (_n(2024, 3, 15, 10), _n(2024, 3, 18)),
            ],
        ),
        _row(
            "ceil_calendar",
            "ceil_1_month",
            [(_n(2024, 1, 1), _n(2024, 2, 1)), (_n(2024, 1, 15), _n(2024, 2, 1)), (_n(2024, 12, 15), _n(2025, 1, 1))],
        ),
        _row(
            "ceil_calendar",
            "ceil_1_year",
            [
                (_n(2024, 1, 1), _n(2025, 1, 1)),
                (_n(2023, 12, 31, 23, 59, 59), _n(2024, 1, 1)),
                (_n(2024, 6, 15), _n(2025, 1, 1)),
            ],
        ),
        # round: ties go up
        _row(
            "round_tie",
            "round_1_minute",
            [
                (_n(2024, 3, 15, 10, 20, 29), _n(2024, 3, 15, 10, 20)),
                (_n(2024, 3, 15, 10, 20, 30), _n(2024, 3, 15, 10, 21)),
                (_n(2024, 3, 15, 10, 20, 45), _n(2024, 3, 15, 10, 21)),
            ],
        ),
        _row(
            "round_tie",
            "round_1_hour",
            [
                (_n(2024, 3, 15, 10, 29, 59), _n(2024, 3, 15, 10)),
                (_n(2024, 3, 15, 10, 30), _n(2024, 3, 15, 11)),
                (_n(2024, 3, 15, 23, 45), _n(2024, 3, 16)),
            ],
        ),
        _row(
            "round_tie",
            "round_1_day",
            [
                (_n(2024, 3, 15, 11, 59, 59), _n(2024, 3, 15)),
                (_n(2024, 3, 15, 12), _n(2024, 3, 16)),
                (_n(2024, 12, 31, 18), _n(2025, 1, 1)),
            ],
        ),
        _row(
            "round_tie",
            "round_10_minute",
            [
                (_n(2024, 6, 15, 14, 24, 59), _n(2024, 6, 15, 14, 20)),
                (_n(2024, 6, 15, 14, 25), _n(2024, 6, 15, 14, 30)),
                (_n(2024, 6, 15, 14, 35), _n(2024, 6, 15, 14, 40)),
            ],
        ),
        _row(
            "round_tie",
            "round_1_week",
            [
                (_n(2024, 3, 14, 11, 59), _n(2024, 3, 11)),
                (_n(2024, 3, 14, 12), _n(2024, 3, 18)),
                (_n(2024, 3, 17, 18), _n(2024, 3, 18)),
            ],
        ),
        # round: the length of the month or year
        _row(
            "round_length",
            "round_1_month",
            [
                (_n(2024, 1, 16, 11, 59, 59), _n(2024, 1, 1)),
                (_n(2024, 1, 16, 12), _n(2024, 2, 1)),
                (_n(2024, 12, 16, 12), _n(2025, 1, 1)),
            ],
        ),
        _row(
            "round_length",
            "round_1_month",
            [
                (_n(2023, 2, 14, 23, 59), _n(2023, 2, 1)),
                (_n(2023, 2, 15), _n(2023, 3, 1)),
                (_n(2024, 2, 15), _n(2024, 2, 1)),
                (_n(2024, 2, 15, 12), _n(2024, 3, 1)),
            ],
        ),
        _row(
            "round_length",
            "round_1_year",
            [
                (_n(2023, 7, 2, 11, 59, 59), _n(2023, 1, 1)),
                (_n(2023, 7, 2, 12), _n(2024, 1, 1)),
                (_n(2024, 7, 1, 23, 59, 59), _n(2024, 1, 1)),
                (_n(2024, 7, 2), _n(2025, 1, 1)),
            ],
        ),
        # leap day
        _row("leap_day", "floor_1_day", [(_n(2024, 2, 29, 13, 45), _n(2024, 2, 29))]),
        _row(
            "leap_day", "ceil_1_day", [(_n(2024, 2, 29, 13, 45), _n(2024, 3, 1)), (_n(2023, 2, 28, 12), _n(2023, 3, 1))]
        ),
        _row("leap_day", "floor_1_month", [(_n(2024, 2, 29, 13, 45), _n(2024, 2, 1))]),
        _row("leap_day", "ceil_1_month", [(_n(2024, 2, 29, 13, 45), _n(2024, 3, 1))]),
        _row("leap_day", "ceil_1_year", [(_n(2024, 2, 29, 13, 45), _n(2025, 1, 1))]),
        _row("leap_day", "round_1_month", [(_n(2024, 2, 29, 12), _n(2024, 3, 1))]),
        # round on a UTC column is plain wall-clock half-up rounding
        _row(
            "round_length",
            "round_1_month",
            [(_u(2024, 1, 16, 12), _u(2024, 2, 1)), (_u(2024, 1, 16, 11), _u(2024, 1, 1))],
            _AWARE,
        ),
        _row(
            "row_order",
            "floor_1_day",
            [
                (_n(2024, 3, 15, 10), _n(2024, 3, 15)),
                (_n(2023, 1, 1, 5), _n(2023, 1, 1)),
                (_n(2024, 3, 14, 23), _n(2024, 3, 14)),
            ],
        ),
    ),
)
