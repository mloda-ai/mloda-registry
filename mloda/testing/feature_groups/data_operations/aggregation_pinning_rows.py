"""Hand-computed pinning rows of the aggregation, window, frame and scalar aggregate references."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from types import MappingProxyType

from mloda.user import DataType

from mloda.testing.feature_groups.data_operations.pinning import PinningRow, PinningTable

NAN = float("nan")

AGGREGATION_RULES: Mapping[str, str] = MappingProxyType(
    {
        "nan": "NaN is a value: nunique counts it once, min and max skip it, first keeps a leading NaN.",
        "mean": "mean is avg.",
        "signed_zero": "0.0 and -0.0 are one partition key and one nunique value; the key keeps its first-seen sign.",
        "partition_keys": "NaN partition keys form one group; groups come in first-occurrence order.",
        "mask": "A row failing the mask is skipped; a group with no row left aggregates to None.",
        "none": "A None source value is skipped by the aggregate.",
        "order": "order_by sorts real values ascending, then NaN, then None.",
        "ties": "Rows tied on order_by keep their input order.",
        "time_window": "A time frame holds the rows up to the current one whose stamp is not before the window start.",
        "dtype": "sum of an unsigned or bool column widens to UINT64.",
    }
)

_I = DataType.INT64
_D = DataType.DOUBLE
_S = DataType.STRING
_TS = DataType.TIMESTAMP_MICROS_TZ

_BY_G = {"partition_by": ["g"]}
_BY_G_ORDER_O = {"partition_by": ["g"], "order_by": "o"}
_BY_REGION_ORDER_TS = {"partition_by": ["region"], "order_by": "ts"}


def _utc(year: int, month: int, day: int, hour: int = 0) -> datetime:
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


_TIED_SMALL = {
    "g": (_S, ("a",) * 3),
    "o": (_TS, (_utc(2023, 1, 2), _utc(2023, 1, 2), _utc(2023, 1, 1))),
    "v": (_I, (1, 2, 3)),
}
_TIED_LONG = {"g": (_S, ("a",) * 25), "o": (_TS, (_utc(2023, 1, 1),) * 25), "v": (_I, tuple(range(25)))}


def _calendar_row(stamps: tuple[datetime, ...], unit: str) -> PinningRow:
    """A 1-unit calendar frame over rows holding 10**i, so each sum names its members: the 2nd row is in, the 1st out."""
    feature = f"value__sum_1_{unit}_window"
    return PinningRow(
        case="time_window",
        feature=feature,
        inputs={"region": (_S, ("A",) * 3), "ts": (_TS, stamps), "value": (_I, (1, 10, 100))},
        expected={feature: (_I, (1, 11, 110))},
        context=_BY_REGION_ORDER_TS,
    )


AGGREGATION_PINNING = PinningTable(
    operation="aggregation",
    rules=AGGREGATION_RULES,
    row_preserving=False,
    rows=(
        PinningRow(
            case="nan",
            feature="v__nunique_agg",
            inputs={"g": (_S, ("a",) * 4), "v": (_D, (NAN, NAN, 1.0, None))},
            expected={"g": (_S, ("a",)), "v__nunique_agg": (_I, (2,))},
            context=_BY_G,
        ),
        PinningRow(
            case="nan",
            feature="val__first_agg",
            inputs={"grp": (_S, ("a", "a")), "val": (_D, (NAN, 1.0))},
            expected={"grp": (_S, ("a",)), "val__first_agg": (_D, (NAN,))},
            context={"partition_by": ["grp"]},
        ),
        PinningRow(
            case="nan",
            feature="val__min_agg",
            inputs={"grp": (_S, ("A",) * 3), "val": (_D, (NAN, 1.0, 3.0))},
            expected={"grp": (_S, ("A",)), "val__min_agg": (_D, (1.0,))},
            context={"partition_by": ["grp"]},
        ),
        PinningRow(
            case="nan",
            feature="val__max_agg",
            inputs={"grp": (_S, ("A",) * 3), "val": (_D, (NAN, 1.0, 3.0))},
            expected={"grp": (_S, ("A",)), "val__max_agg": (_D, (3.0,))},
            context={"partition_by": ["grp"]},
        ),
        PinningRow(
            case="mean",
            feature="v__mean_agg",
            inputs={"g": (_S, ("a",) * 4), "v": (_D, (1.0, 2.0, None, 6.0))},
            expected={"g": (_S, ("a",)), "v__mean_agg": (_D, (3.0,))},
            context=_BY_G,
        ),
        PinningRow(
            case="mean",
            feature="v__avg_agg",
            inputs={"g": (_S, ("a",) * 4), "v": (_D, (1.0, 2.0, None, 6.0))},
            expected={"g": (_S, ("a",)), "v__avg_agg": (_D, (3.0,))},
            context=_BY_G,
        ),
        PinningRow(
            case="signed_zero",
            feature="v__count_agg",
            inputs={"k": (_D, (-0.0, 0.0, 0.0)), "v": (_I, (1, 2, 3))},
            expected={"k": (_D, (-0.0,)), "v__count_agg": (_I, (3,))},
            context={"partition_by": ["k"]},
        ),
        PinningRow(
            case="signed_zero",
            feature="v__nunique_agg",
            inputs={"k": (_S, ("a",) * 3), "v": (_D, (0.0, -0.0, -0.0))},
            expected={"k": (_S, ("a",)), "v__nunique_agg": (_I, (1,))},
            context={"partition_by": ["k"]},
        ),
        PinningRow(
            case="partition_keys",
            feature="val__sum_agg",
            inputs={"grp": (_D, (NAN, NAN, 1.0, 2.0)), "val": (_D, (10.0, 20.0, 100.0, 200.0))},
            expected={"grp": (_D, (NAN, 1.0, 2.0)), "val__sum_agg": (_D, (30.0, 100.0, 200.0))},
            context={"partition_by": ["grp"]},
        ),
        PinningRow(
            case="mask",
            feature="value__sum_agg",
            inputs={
                "g": (_S, ("a", "a", "b", "b", "c")),
                "c": (_S, ("X", "Y", "X", "X", "Y")),
                "value": (_I, (1, 10, 2, 3, 100)),
            },
            expected={"g": (_S, ("a", "b", "c")), "value__sum_agg": (_I, (1, 5, None))},
            context={"partition_by": ["g"], "mask": ("c", "equal", "X")},
        ),
        PinningRow(
            case="dtype",
            feature="v__sum_agg",
            inputs={"g": (_S, ("a", "a")), "v": (DataType.UINT8, (200, 100))},
            expected={"g": (_S, ("a",)), "v__sum_agg": (DataType.UINT64, (300,))},
            context=_BY_G,
        ),
    ),
)

WINDOW_AGGREGATION_PINNING = PinningTable(
    operation="window_aggregation",
    rules=AGGREGATION_RULES,
    rows=(
        PinningRow(
            case="partition_keys",
            feature="val__sum_window",
            inputs={"grp": (_D, (NAN, NAN, 1.0, 2.0)), "val": (_D, (10.0, 20.0, 100.0, 200.0))},
            expected={"val__sum_window": (_D, (30.0, 30.0, 100.0, 200.0))},
            context={"partition_by": ["grp"]},
        ),
        PinningRow(
            case="nan",
            feature="val__min_window",
            inputs={"grp": (_S, ("A",) * 3), "val": (_D, (NAN, 1.0, 3.0))},
            expected={"val__min_window": (_D, (1.0, 1.0, 1.0))},
            context={"partition_by": ["grp"]},
        ),
        PinningRow(
            case="nan",
            feature="val__max_window",
            inputs={"grp": (_S, ("A",) * 3), "val": (_D, (NAN, 1.0, 3.0))},
            expected={"val__max_window": (_D, (3.0, 3.0, 3.0))},
            context={"partition_by": ["grp"]},
        ),
        PinningRow(
            case="nan",
            feature="v__first_window",
            inputs={"g": (_S, ("a", "a")), "o": (_D, (1.0, 2.0)), "v": (_D, (NAN, 1.0))},
            expected={"v__first_window": (_D, (NAN, NAN))},
            context=_BY_G_ORDER_O,
        ),
        PinningRow(
            case="signed_zero",
            feature="v__count_window",
            inputs={"k": (_D, (0.0, -0.0, -0.0)), "v": (_I, (1, 2, 3))},
            expected={"v__count_window": (_I, (3, 3, 3))},
            context={"partition_by": ["k"]},
        ),
        PinningRow(
            case="signed_zero",
            feature="v__nunique_window",
            inputs={"k": (_S, ("a",) * 3), "v": (_D, (0.0, -0.0, -0.0))},
            expected={"v__nunique_window": (_I, (1, 1, 1))},
            context={"partition_by": ["k"]},
        ),
        PinningRow(
            case="order",
            feature="v__last_window",
            inputs={"g": (_S, ("a",) * 3), "o": (_D, (None, NAN, 1.0)), "v": (_I, (10, 20, 30))},
            expected={"v__last_window": (_I, (10, 10, 10))},
            context=_BY_G_ORDER_O,
        ),
        PinningRow(
            case="order",
            feature="v__first_window",
            inputs={"g": (_S, ("a",) * 2), "o": (_D, (None, NAN)), "v": (_I, (10, 20))},
            expected={"v__first_window": (_I, (20, 20))},
            context=_BY_G_ORDER_O,
        ),
    ),
)

FRAME_AGGREGATE_PINNING = PinningTable(
    operation="frame_aggregate",
    rules=AGGREGATION_RULES,
    rows=(
        PinningRow(
            case="time_window",
            feature="my_time_sum",
            inputs={
                "region": (_S, ("A",) * 4),
                "ts": (_TS, (_utc(2023, 1, 1), _utc(2023, 1, 2), _utc(2023, 1, 4), _utc(2023, 1, 5))),
                "value": (_I, (10, 20, 30, 40)),
            },
            expected={"my_time_sum": (_I, (10, 30, 50, 70))},
            context={
                "aggregation_type": "sum",
                "frame_type": "time",
                "frame_size": 2,
                "frame_unit": "day",
                "in_features": "value",
                **_BY_REGION_ORDER_TS,
            },
        ),
        PinningRow(
            case="time_window",
            feature="value__sum_3_day_window",
            inputs={
                "region": (_S, ("A",) * 3),
                "ts": (_TS, (_utc(2023, 1, 1), _utc(2023, 1, 1), _utc(2023, 1, 6))),
                "value": (_I, (10, 20, 30)),
            },
            expected={"value__sum_3_day_window": (_I, (10, 30, 30))},
            context=_BY_REGION_ORDER_TS,
        ),
        _calendar_row((_utc(2023, 2, 27), _utc(2023, 2, 28), _utc(2023, 3, 31)), "month"),
        _calendar_row((_utc(2024, 2, 28), _utc(2024, 2, 29), _utc(2024, 3, 31)), "month"),
        _calendar_row((_utc(2023, 2, 27), _utc(2023, 2, 28), _utc(2024, 2, 29)), "year"),
        _calendar_row((_utc(2023, 2, 28, 11), _utc(2023, 2, 28, 13), _utc(2023, 3, 31, 12)), "month"),
        PinningRow(
            case="none",
            feature="value__sum_rolling_3",
            inputs={"region": (_S, ("A",) * 4), "ts": (_I, (1, 2, 3, 4)), "value": (_I, (10, None, 30, 40))},
            expected={"value__sum_rolling_3": (_I, (10, 10, 40, 70))},
            context=_BY_REGION_ORDER_TS,
        ),
        PinningRow(
            case="order",
            feature="value__cumsum",
            inputs={"region": (_S, ("A",) * 3), "ts": (_I, (None, 1, 2)), "value": (_I, (100, 10, 20))},
            expected={"value__cumsum": (_I, (130, 10, 30))},
            context=_BY_REGION_ORDER_TS,
        ),
        PinningRow(
            case="order",
            feature="v__cumsum",
            inputs={"g": (_S, ("a",) * 3), "o": (_D, (None, NAN, 1.0)), "v": (_I, (10, 20, 30))},
            expected={"v__cumsum": (_I, (60, 50, 30))},
            context=_BY_G_ORDER_O,
        ),
        PinningRow(
            case="ties",
            feature="v__cumsum",
            inputs=_TIED_SMALL,
            expected={"v__cumsum": (_I, (4, 6, 3))},
            context=_BY_G_ORDER_O,
        ),
        PinningRow(
            case="ties",
            feature="v__sum_rolling_2",
            inputs=_TIED_SMALL,
            expected={"v__sum_rolling_2": (_I, (4, 3, 3))},
            context=_BY_G_ORDER_O,
        ),
        PinningRow(
            case="ties",
            feature="v__sum_1_day_window",
            inputs=_TIED_SMALL,
            expected={"v__sum_1_day_window": (_I, (4, 6, 3))},
            context=_BY_G_ORDER_O,
        ),
        PinningRow(
            case="ties",
            feature="v__cumsum",
            inputs=_TIED_LONG,
            expected={"v__cumsum": (_I, tuple(i * (i + 1) // 2 for i in range(25)))},
            context=_BY_G_ORDER_O,
        ),
        PinningRow(
            case="ties",
            feature="v__expanding_sum",
            inputs=_TIED_LONG,
            expected={"v__expanding_sum": (_I, tuple(i * (i + 1) // 2 for i in range(25)))},
            context=_BY_G_ORDER_O,
        ),
        PinningRow(
            case="ties",
            feature="v__sum_rolling_3",
            inputs=_TIED_LONG,
            expected={"v__sum_rolling_3": (_I, tuple(sum(range(max(0, i - 2), i + 1)) for i in range(25)))},
            context=_BY_G_ORDER_O,
        ),
    ),
)

SCALAR_AGGREGATE_PINNING = PinningTable(
    operation="scalar_aggregate",
    rules=AGGREGATION_RULES,
    rows=(
        PinningRow(
            case="dtype",
            feature="v__sum_scalar",
            inputs={"v": (DataType.BOOLEAN, (True, False, True))},
            expected={"v__sum_scalar": (DataType.UINT64, (2, 2, 2))},
        ),
    ),
)
