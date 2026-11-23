"""Hand-computed pinning rows of the resample reference."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from types import MappingProxyType

from mloda.user import DataType

from mloda.testing.feature_groups.data_operations.pinning import PinningRow, PinningTable

NAN = float("nan")

RESAMPLE_RULES: Mapping[str, str] = MappingProxyType(
    {
        "bucket": "The time column floors to the start of its n-minute, n-hour or n-day bucket counted from the epoch, UTC; pre-1970 values floor downward.",
        "rows": "One row per non-empty (partition, bucket) pair in first-occurrence order; gap buckets give no row.",
        "none": "None values are skipped; a bucket whose values are all None gives count 0 and None for sum, mean, min and max.",
        "nan": "NaN propagates through sum and mean, count counts it, min and max skip it, and an all-NaN bucket gives NaN.",
        "null_time": "A None timestamp is its own bucket, with or without partition_by.",
        "time_source": "When the source is the time column, min and max aggregate floored bucket starts; count counts non-null timestamps.",
        "partition_keys": "NaN partition keys merge, a None key is its own group, 0.0 and -0.0 merge and keep the first-seen value.",
        "whole_table": "Without partition_by the buckets alone group the rows.",
        "dtype": "count is int64, mean is float64, sum widens ints to int64, uint64 to uint64 and floats to float64, min and max keep the source type.",
        "empty": "An empty input gives empty columns: the partition columns, the time column, then the feature.",
    }
)

_I = DataType.INT64
_D = DataType.DOUBLE
_S = DataType.STRING
_TS = DataType.TIMESTAMP_MICROS_TZ
_CTX = {"partition_by": ["p"], "time_column": "ts"}
_WHOLE = {"time_column": "ts"}


def _t(hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2023, 1, 1, hour, minute, second, tzinfo=timezone.utc)


def _row(
    case: str,
    feature: str,
    inputs: dict[str, tuple[DataType, tuple[object, ...]]],
    expected: dict[str, tuple[DataType, tuple[object, ...]]],
    context: Mapping[str, object] = _CTX,
) -> PinningRow:
    return PinningRow(case=case, feature=feature, inputs=inputs, expected=expected, context=context)


_A4 = ("a",) * 4

RESAMPLE_PINNING = PinningTable(
    operation="resample",
    rules=RESAMPLE_RULES,
    row_preserving=False,
    rows=(
        _row(
            "time_source",
            "ts__resample_1_hour_min",
            {"p": (_S, ("a",) * 3), "ts": (_TS, (_t(8, 10), _t(8, 50), None))},
            {"p": (_S, ("a", "a")), "ts": (_TS, (_t(8), None)), "ts__resample_1_hour_min": (_TS, (_t(8), None))},
        ),
        _row(
            "time_source",
            "ts__resample_1_hour_max",
            {"ts": (_TS, (_t(8, 10), _t(8, 50), _t(9, 20)))},
            {"ts": (_TS, (_t(8), _t(9))), "ts__resample_1_hour_max": (_TS, (_t(8), _t(9)))},
            _WHOLE,
        ),
        _row(
            "time_source",
            "ts__resample_1_hour_count",
            {"p": (_S, ("a",) * 3), "ts": (_TS, (_t(8, 10), _t(8, 50), None))},
            {"p": (_S, ("a", "a")), "ts": (_TS, (_t(8), None)), "ts__resample_1_hour_count": (_I, (2, 0))},
        ),
        _row(
            "bucket",
            "v__resample_15_minute_mean",
            {"p": (_S, _A4), "ts": (_TS, (_t(8, 5), _t(8, 20), _t(8, 15), _t(8, 29))), "v": (_I, (1, 2, 4, 6))},
            # 08:05 floors to 08:00; 08:20, 08:15 (on the boundary) and 08:29 floor to 08:15: (2 + 4 + 6) / 3
            {"p": (_S, ("a", "a")), "ts": (_TS, (_t(8), _t(8, 15))), "v__resample_15_minute_mean": (_D, (1.0, 4.0))},
        ),
        _row(
            "bucket",
            "v__resample_1_hour_sum",
            {"p": (_S, ("a", "a")), "ts": (_TS, (_t(8), _t(7, 59, 59))), "v": (_I, (1, 2))},
            # 08:00:00 is on the boundary and starts its own bucket; 07:59:59 belongs to 07:00
            {"p": (_S, ("a", "a")), "ts": (_TS, (_t(8), _t(7))), "v__resample_1_hour_sum": (_I, (1, 2))},
        ),
        _row(
            "bucket",
            "v__resample_2_hour_count",
            {"p": (_S, ("a",) * 3), "ts": (_TS, (_t(3, 30), _t(4, 10), _t(5, 59))), "v": (_I, (1, 2, 3))},
            # 2-hour buckets start at even hours: 03:30 is in 02:00, 04:10 and 05:59 are in 04:00
            {"p": (_S, ("a", "a")), "ts": (_TS, (_t(2), _t(4))), "v__resample_2_hour_count": (_I, (1, 2))},
        ),
        _row(
            "bucket",
            "v__resample_1_day_sum",
            {
                "p": (_S, ("a",) * 3),
                "ts": (
                    _TS,
                    (
                        datetime(1969, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
                        datetime(1969, 12, 31, 0, 0, 1, tzinfo=timezone.utc),
                        datetime(1970, 1, 1, tzinfo=timezone.utc),
                    ),
                ),
                "v": (_I, (3, 4, 5)),
            },
            # before 1970 the day floors downward: both 1969-12-31 values land in 1969-12-31 (3 + 4)
            {
                "p": (_S, ("a", "a")),
                "ts": (_TS, (datetime(1969, 12, 31, tzinfo=timezone.utc), datetime(1970, 1, 1, tzinfo=timezone.utc))),
                "v__resample_1_day_sum": (_I, (7, 5)),
            },
        ),
        _row(
            "bucket",
            "v__resample_1_hour_sum",
            {
                "p": (_S, ("a",) * 3),
                "ts": (
                    DataType.TIMESTAMP_MILLIS,
                    (datetime(2023, 1, 1, 8, 10), datetime(2023, 1, 1, 8, 50), datetime(2023, 1, 1, 9, 0)),
                ),
                "v": (_I, (1, 2, 4)),
            },
            # a millisecond naive time column keeps its type: 08:10 and 08:50 floor to 08:00 (1 + 2), 09:00 is on the boundary
            {
                "p": (_S, ("a", "a")),
                "ts": (DataType.TIMESTAMP_MILLIS, (datetime(2023, 1, 1, 8), datetime(2023, 1, 1, 9))),
                "v__resample_1_hour_sum": (_I, (3, 4)),
            },
        ),
        _row(
            "bucket",
            "v__resample_2_day_sum",
            {
                "p": (_S, _A4),
                "ts": (
                    _TS,
                    (
                        datetime(1969, 12, 30, 12, tzinfo=timezone.utc),
                        datetime(1969, 12, 31, 12, tzinfo=timezone.utc),
                        datetime(1970, 1, 1, 12, tzinfo=timezone.utc),
                        datetime(1970, 1, 2, 12, tzinfo=timezone.utc),
                    ),
                ),
                "v": (_I, (1, 2, 4, 8)),
            },
            # 2-day buckets count from the epoch (day 0 = 1970-01-01): days -2 and -1 floor to day -2 (1969-12-30), days 0 and 1 to day 0
            {
                "p": (_S, ("a", "a")),
                "ts": (_TS, (datetime(1969, 12, 30, tzinfo=timezone.utc), datetime(1970, 1, 1, tzinfo=timezone.utc))),
                "v__resample_2_day_sum": (_I, (3, 12)),
            },
        ),
        _row(
            "bucket",
            "v__resample_7_minute_sum",
            {
                "p": (_S, _A4),
                "ts": (_TS, (_t(8, 1), _t(8, 2), _t(8, 8), _t(8, 9))),
                "v": (_I, (1, 2, 4, 8)),
            },
            # 2023-01-01 is day 19358 = 27875520 minutes, and 27875520 % 7 = 1, so boundaries fall at 07:55, 08:02, 08:09 (08:00 is not one)
            {
                "p": (_S, ("a", "a", "a")),
                "ts": (_TS, (_t(7, 55), _t(8, 2), _t(8, 9))),
                "v__resample_7_minute_sum": (_I, (1, 6, 8)),
            },
        ),
        _row(
            "rows",
            "v__resample_1_hour_sum",
            {
                "p": (_S, ("a", "b", "a", "b", "a")),
                "ts": (_TS, (_t(9, 10), _t(8, 20), _t(9, 40), _t(8, 30), _t(12, 5))),
                "v": (_I, (1, 2, 3, 4, 5)),
            },
            # groups in first-occurrence order: (a, 09:00) 1 + 3, (b, 08:00) 2 + 4, (a, 12:00) 5; no rows for 10:00 or 11:00
            {
                "p": (_S, ("a", "b", "a")),
                "ts": (_TS, (_t(9), _t(8), _t(12))),
                "v__resample_1_hour_sum": (_I, (4, 6, 5)),
            },
        ),
        _row(
            "none",
            "v__resample_1_hour_mean",
            {"p": (_S, ("a",) * 3), "ts": (_TS, (_t(8, 10), _t(8, 20), _t(8, 30))), "v": (_I, (None, 2, 6))},
            # None skipped: (2 + 6) / 2
            {"p": (_S, ("a",)), "ts": (_TS, (_t(8),)), "v__resample_1_hour_mean": (_D, (4.0,))},
        ),
        _row(
            "none",
            "v__resample_1_hour_count",
            {"p": (_S, ("a",) * 3), "ts": (_TS, (_t(8, 10), _t(8, 20), _t(8, 30))), "v": (_I, (1, None, 3))},
            {"p": (_S, ("a",)), "ts": (_TS, (_t(8),)), "v__resample_1_hour_count": (_I, (2,))},
        ),
        *(
            _row(
                "none",
                f"v__resample_1_hour_{agg}",
                {
                    "p": (_S, ("a", "a", "b")),
                    "ts": (_TS, (_t(8, 10), _t(8, 20), _t(8, 30))),
                    "v": (_I, (None, None, 5)),
                },
                # group a is an all-None bucket: it still emits a row, with None
                {
                    "p": (_S, ("a", "b")),
                    "ts": (_TS, (_t(8), _t(8))),
                    f"v__resample_1_hour_{agg}": (out_type, (None, five)),
                },
            )
            for agg, out_type, five in (("sum", _I, 5), ("mean", _D, 5.0), ("min", _I, 5), ("max", _I, 5))
        ),
        _row(
            "none",
            "v__resample_1_hour_count",
            {"p": (_S, ("a", "a", "b")), "ts": (_TS, (_t(8, 10), _t(8, 20), _t(8, 30))), "v": (_I, (None, None, 5))},
            {"p": (_S, ("a", "b")), "ts": (_TS, (_t(8), _t(8))), "v__resample_1_hour_count": (_I, (0, 1))},
        ),
        _row(
            "nan",
            "v__resample_1_hour_sum",
            {"p": (_S, ("a",) * 3), "ts": (_TS, (_t(8, 10), _t(8, 20), _t(8, 30))), "v": (_D, (1.0, NAN, None))},
            {"p": (_S, ("a",)), "ts": (_TS, (_t(8),)), "v__resample_1_hour_sum": (_D, (NAN,))},
        ),
        _row(
            "nan",
            "v__resample_1_hour_mean",
            {"p": (_S, ("a",) * 2), "ts": (_TS, (_t(8, 10), _t(8, 20))), "v": (_D, (1.0, NAN))},
            {"p": (_S, ("a",)), "ts": (_TS, (_t(8),)), "v__resample_1_hour_mean": (_D, (NAN,))},
        ),
        _row(
            "nan",
            "v__resample_1_hour_count",
            {"p": (_S, ("a",) * 3), "ts": (_TS, (_t(8, 10), _t(8, 20), _t(8, 30))), "v": (_D, (1.0, NAN, None))},
            # NaN is counted, None is not
            {"p": (_S, ("a",)), "ts": (_TS, (_t(8),)), "v__resample_1_hour_count": (_I, (2,))},
        ),
        _row(
            "nan",
            "v__resample_1_hour_min",
            {"p": (_S, ("a",) * 3), "ts": (_TS, (_t(8, 10), _t(8, 20), _t(8, 30))), "v": (_D, (3.0, NAN, 1.0))},
            {"p": (_S, ("a",)), "ts": (_TS, (_t(8),)), "v__resample_1_hour_min": (_D, (1.0,))},
        ),
        _row(
            "nan",
            "v__resample_1_hour_max",
            {"p": (_S, ("a",) * 3), "ts": (_TS, (_t(8, 10), _t(8, 20), _t(8, 30))), "v": (_D, (3.0, NAN, 1.0))},
            {"p": (_S, ("a",)), "ts": (_TS, (_t(8),)), "v__resample_1_hour_max": (_D, (3.0,))},
        ),
        _row(
            "nan",
            "v__resample_1_hour_min",
            {"p": (_S, ("a",) * 2), "ts": (_TS, (_t(8, 10), _t(8, 20))), "v": (_D, (NAN, NAN))},
            # every value is NaN: nothing to skip to, so NaN
            {"p": (_S, ("a",)), "ts": (_TS, (_t(8),)), "v__resample_1_hour_min": (_D, (NAN,))},
        ),
        _row(
            "nan",
            "v__resample_1_hour_max",
            {"p": (_S, ("a",) * 2), "ts": (_TS, (_t(8, 10), _t(8, 20))), "v": (_D, (NAN, NAN))},
            {"p": (_S, ("a",)), "ts": (_TS, (_t(8),)), "v__resample_1_hour_max": (_D, (NAN,))},
        ),
        _row(
            "null_time",
            "v__resample_1_hour_sum",
            {"p": (_S, _A4), "ts": (_TS, (_t(8, 10), None, _t(8, 50), None)), "v": (_I, (1, 2, 4, 8))},
            # the None timestamp is its own bucket: (a, 08:00) 1 + 4, (a, None) 2 + 8
            {"p": (_S, ("a", "a")), "ts": (_TS, (_t(8), None)), "v__resample_1_hour_sum": (_I, (5, 10))},
        ),
        _row(
            "null_time",
            "v__resample_1_hour_sum",
            {"ts": (_TS, (_t(9, 10), None, None)), "v": (_I, (2, 1, 4))},
            {"ts": (_TS, (_t(9), None)), "v__resample_1_hour_sum": (_I, (2, 5))},
            _WHOLE,
        ),
        _row(
            "partition_keys",
            "v__resample_1_hour_sum",
            {"p": (_D, (NAN, 1.0, NAN)), "ts": (_TS, (_t(8, 10),) * 3), "v": (_I, (1, 2, 4))},
            # the two NaN keys merge: 1 + 4
            {"p": (_D, (NAN, 1.0)), "ts": (_TS, (_t(8), _t(8))), "v__resample_1_hour_sum": (_I, (5, 2))},
        ),
        _row(
            "partition_keys",
            "v__resample_1_hour_sum",
            {"p": (_D, (None, NAN, None, NAN)), "ts": (_TS, (_t(8, 10),) * 4), "v": (_I, (1, 2, 4, 8))},
            # None and NaN are different keys: None 1 + 4, NaN 2 + 8
            {"p": (_D, (None, NAN)), "ts": (_TS, (_t(8), _t(8))), "v__resample_1_hour_sum": (_I, (5, 10))},
        ),
        _row(
            "partition_keys",
            "v__resample_1_hour_count",
            {"p": (_D, (0.0, -0.0, -0.0)), "ts": (_TS, (_t(8, 10),) * 3), "v": (_I, (1, 2, 4))},
            # 0.0 and -0.0 merge; the key keeps the first-seen 0.0
            {"p": (_D, (0.0,)), "ts": (_TS, (_t(8),)), "v__resample_1_hour_count": (_I, (3,))},
        ),
        _row(
            "partition_keys",
            "v__resample_1_hour_count",
            {"p": (_D, (-0.0, 0.0)), "ts": (_TS, (_t(8, 10),) * 2), "v": (_I, (1, 2))},
            # the key keeps the first-seen -0.0
            {"p": (_D, (-0.0,)), "ts": (_TS, (_t(8),)), "v__resample_1_hour_count": (_I, (2,))},
        ),
        _row(
            "whole_table",
            "v__resample_1_hour_sum",
            {"ts": (_TS, (_t(8, 10), _t(8, 50), _t(9, 5))), "v": (_I, (1, 2, 4))},
            {"ts": (_TS, (_t(8), _t(9))), "v__resample_1_hour_sum": (_I, (3, 4))},
            _WHOLE,
        ),
        _row(
            "dtype",
            "v__resample_1_hour_count",
            {"p": (_S, ("a",) * 3), "ts": (_TS, (_t(8, 10), _t(8, 20), _t(8, 30))), "v": (_S, ("x", None, "y"))},
            {"p": (_S, ("a",)), "ts": (_TS, (_t(8),)), "v__resample_1_hour_count": (_I, (2,))},
        ),
        _row(
            "dtype",
            "v__resample_1_hour_mean",
            {"p": (_S, ("a",) * 2), "ts": (_TS, (_t(8, 10), _t(8, 20))), "v": (_I, (1, 2))},
            # mean of an int source is a double
            {"p": (_S, ("a",)), "ts": (_TS, (_t(8),)), "v__resample_1_hour_mean": (_D, (1.5,))},
        ),
        _row(
            "dtype",
            "v__resample_1_hour_sum",
            {"p": (_S, ("a",) * 2), "ts": (_TS, (_t(8, 10), _t(8, 20))), "v": (DataType.INT32, (3, 4))},
            # a signed int sum widens to int64
            {"p": (_S, ("a",)), "ts": (_TS, (_t(8),)), "v__resample_1_hour_sum": (_I, (7,))},
        ),
        _row(
            "dtype",
            "v__resample_1_hour_sum",
            {"p": (_S, ("a",) * 2), "ts": (_TS, (_t(8, 10), _t(8, 20))), "v": (DataType.UINT64, (3, 4))},
            {"p": (_S, ("a",)), "ts": (_TS, (_t(8),)), "v__resample_1_hour_sum": (DataType.UINT64, (7,))},
        ),
        _row(
            "dtype",
            "v__resample_1_hour_sum",
            {"p": (_S, ("a",) * 2), "ts": (_TS, (_t(8, 10), _t(8, 20))), "v": (DataType.FLOAT, (1.5, 2.0))},
            # a float sum is a double
            {"p": (_S, ("a",)), "ts": (_TS, (_t(8),)), "v__resample_1_hour_sum": (_D, (3.5,))},
        ),
        _row(
            "dtype",
            "v__resample_1_hour_min",
            {"p": (_S, ("a",) * 2), "ts": (_TS, (_t(8, 10), _t(8, 20))), "v": (DataType.INT32, (5, 3))},
            {"p": (_S, ("a",)), "ts": (_TS, (_t(8),)), "v__resample_1_hour_min": (DataType.INT32, (3,))},
        ),
        _row(
            "dtype",
            "v__resample_1_hour_max",
            {"p": (_S, ("a",) * 3), "ts": (_TS, (_t(8, 10), _t(8, 20), _t(8, 30))), "v": (_S, ("b", "a", None))},
            {"p": (_S, ("a",)), "ts": (_TS, (_t(8),)), "v__resample_1_hour_max": (_S, ("b",))},
        ),
        _row(
            "dtype",
            "v__resample_1_hour_count",
            {"p": (DataType.INT32, (1, 1)), "ts": (_TS, (_t(8, 10), _t(8, 20))), "v": (_I, (1, 2))},
            # the partition column keeps its source type
            {"p": (DataType.INT32, (1,)), "ts": (_TS, (_t(8),)), "v__resample_1_hour_count": (_I, (2,))},
        ),
        _row(
            "empty",
            "v__resample_1_hour_sum",
            {"p": (_S, ()), "ts": (DataType.TIMESTAMP_MICROS, ()), "v": (_D, ())},
            {"p": (_S, ()), "ts": (DataType.TIMESTAMP_MICROS, ()), "v__resample_1_hour_sum": (_D, ())},
        ),
    ),
)
