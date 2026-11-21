"""Hand-computed pinning rows of the offset reference."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from mloda.user import DataType

from mloda.testing.feature_groups.data_operations.pinning import PinningRow, PinningTable

NAN = float("nan")

OFFSET_RULES: Mapping[str, str] = MappingProxyType(
    {
        "order": "Each partition is ordered by order_by ascending, stably: input order breaks ties.",
        "order_tiers": "In order_by real values come first, then NaN, then None; NaN and None are separate tiers.",
        "none": "None in a source is a missing value: diff and pct_change give None when either operand is None.",
        "nan": "NaN in a source is a value: lag, lead, first_value and last_value carry it, diff propagates it.",
        "pct_change_zero": "pct_change with a previous value of zero gives None.",
        "first_last": "first_value and last_value skip None, give None for an all-None partition and broadcast.",
        "partition_keys": "NaN partition keys merge into one group; a None partition key is its own group.",
        "dtype": "lag, lead, first_value and last_value keep the source type; diff is int64 for any integer source, float64 for a float source and a duration for a timestamp source; pct_change is float64.",
        "empty": "An empty input gives an empty column of the declared type.",
    }
)

_I = DataType.INT64
_D = DataType.DOUBLE
_S = DataType.STRING
_CTX = {"partition_by": ["p"], "order_by": "o"}

OFFSET_PINNING = PinningTable(
    operation="offset",
    rules=OFFSET_RULES,
    rows=(
        PinningRow(
            case="order",
            feature="v__lead_1_offset",
            inputs={"p": (_S, ("a",) * 3), "o": (_I, (3, 1, 2)), "v": (_I, (10, 20, 30))},
            # order idx 1, 2, 0 holds 20, 30, 10
            expected={"v__lead_1_offset": (_I, (None, 30, 10))},
            context=_CTX,
        ),
        PinningRow(
            case="order",
            feature="v__lag_1_offset",
            inputs={"p": (_S, ("a",) * 3), "o": (_D, (1.0, 1.0, 1.0)), "v": (_I, (10, 20, 30))},
            expected={"v__lag_1_offset": (_I, (None, 10, 20))},
            context=_CTX,
        ),
        PinningRow(
            case="order",
            feature="v__lag_2_offset",
            inputs={"p": (_S, ("a",) * 4), "o": (_S, ("d", "a", "c", "b")), "v": (_I, (1, 2, 3, 4))},
            # order idx 1, 3, 2, 0 holds 2, 4, 3, 1
            expected={"v__lag_2_offset": (_I, (4, None, 2, None))},
            context=_CTX,
        ),
        PinningRow(
            case="order_tiers",
            feature="v__lag_1_offset",
            inputs={"p": (_S, ("a",) * 3), "o": (_D, (None, NAN, 1.0)), "v": (_I, (10, 20, 30))},
            # order idx 2, 1, 0 holds 30, 20, 10
            expected={"v__lag_1_offset": (_I, (20, 30, None))},
            context=_CTX,
        ),
        PinningRow(
            case="order_tiers",
            feature="v__lead_1_offset",
            inputs={"p": (_S, ("a",) * 3), "o": (_D, (None, NAN, 1.0)), "v": (_I, (10, 20, 30))},
            expected={"v__lead_1_offset": (_I, (None, 10, 20))},
            context=_CTX,
        ),
        PinningRow(
            case="none",
            feature="v__lag_1_offset",
            inputs={"p": (_S, ("a",) * 3), "o": (_I, (1, 2, 3)), "v": (_I, (1, None, 3))},
            expected={"v__lag_1_offset": (_I, (None, 1, None))},
            context=_CTX,
        ),
        PinningRow(
            case="none",
            feature="v__diff_1_offset",
            inputs={"p": (_S, ("a",) * 4), "o": (_I, (1, 2, 3, 4)), "v": (_I, (5, None, 9, 12))},
            expected={"v__diff_1_offset": (_I, (None, None, None, 3))},
            context=_CTX,
        ),
        PinningRow(
            case="nan",
            feature="v__lag_1_offset",
            inputs={"p": (_S, ("a",) * 3), "o": (_I, (1, 2, 3)), "v": (_D, (NAN, 1.5, None))},
            expected={"v__lag_1_offset": (_D, (None, NAN, 1.5))},
            context=_CTX,
        ),
        PinningRow(
            case="nan",
            feature="v__diff_1_offset",
            inputs={"p": (_S, ("a",) * 3), "o": (_I, (1, 2, 3)), "v": (_D, (1.5, 4.0, NAN))},
            expected={"v__diff_1_offset": (_D, (None, 2.5, NAN))},
            context=_CTX,
        ),
        PinningRow(
            case="pct_change_zero",
            feature="v__pct_change_1_offset",
            inputs={"p": (_S, ("a",) * 3), "o": (_I, (1, 2, 3)), "v": (_I, (0, 5, 6))},
            expected={"v__pct_change_1_offset": (_D, (None, None, 0.2))},
            context=_CTX,
        ),
        PinningRow(
            case="pct_change_zero",
            feature="v__pct_change_1_offset",
            inputs={"p": (_S, ("a",) * 5), "o": (_I, (1, 2, 3, 4, 5)), "v": (_I, (4, 6, 3, 0, 2))},
            # (6 - 4) / 4, (3 - 6) / 6, (0 - 3) / 3, previous 0
            expected={"v__pct_change_1_offset": (_D, (None, 0.5, -0.5, -1.0, None))},
            context=_CTX,
        ),
        PinningRow(
            case="first_last",
            feature="v__first_value_offset",
            inputs={"p": (_S, ("a",) * 4), "o": (_I, (1, 2, 3, 4)), "v": (_I, (None, 7, None, 9))},
            expected={"v__first_value_offset": (_I, (7, 7, 7, 7))},
            context=_CTX,
        ),
        PinningRow(
            case="first_last",
            feature="v__last_value_offset",
            inputs={"p": (_S, ("a",) * 4), "o": (_I, (4, 3, 2, 1)), "v": (_I, (None, 7, None, 9))},
            # order idx 3, 2, 1, 0 holds 9, None, 7, None
            expected={"v__last_value_offset": (_I, (7, 7, 7, 7))},
            context=_CTX,
        ),
        PinningRow(
            case="first_last",
            feature="v__first_value_offset",
            inputs={"p": (_S, ("a", "a", "b")), "o": (_I, (1, 2, 1)), "v": (_I, (None, None, 3))},
            expected={"v__first_value_offset": (_I, (None, None, 3))},
            context=_CTX,
        ),
        PinningRow(
            case="nan",
            feature="v__first_value_offset",
            inputs={"p": (_S, ("a",) * 3), "o": (_I, (1, 2, 3)), "v": (_D, (NAN, 1.0, 2.0))},
            expected={"v__first_value_offset": (_D, (NAN, NAN, NAN))},
            context=_CTX,
        ),
        PinningRow(
            case="nan",
            feature="v__last_value_offset",
            inputs={"p": (_S, ("a",) * 3), "o": (_I, (1, 2, 3)), "v": (_D, (1.0, NAN, None))},
            expected={"v__last_value_offset": (_D, (NAN, NAN, NAN))},
            context=_CTX,
        ),
        PinningRow(
            case="partition_keys",
            feature="v__lag_1_offset",
            inputs={"p": (_D, (NAN, 1.0, NAN)), "o": (_I, (1, 1, 2)), "v": (_I, (10, 20, 30))},
            expected={"v__lag_1_offset": (_I, (None, None, 10))},
            context=_CTX,
        ),
        PinningRow(
            case="partition_keys",
            feature="v__lag_1_offset",
            inputs={"p": (_D, (None, NAN, None, NAN)), "o": (_I, (1, 1, 2, 2)), "v": (_I, (1, 2, 3, 4))},
            # None and NaN keys are two groups: (idx 0, 2) and (idx 1, 3)
            expected={"v__lag_1_offset": (_I, (None, None, 1, 2))},
            context=_CTX,
        ),
        PinningRow(
            case="partition_keys",
            feature="v__lag_1_offset",
            inputs={"p": (_D, (0.0, -0.0, 0.0)), "o": (_I, (1, 2, 3)), "v": (_I, (1, 2, 3))},
            expected={"v__lag_1_offset": (_I, (None, 1, 2))},
            context=_CTX,
        ),
        PinningRow(
            case="dtype",
            feature="v__lag_1_offset",
            inputs={"p": (_S, ("a", "a")), "o": (_I, (1, 2)), "v": (_S, ("x", "y"))},
            expected={"v__lag_1_offset": (_S, (None, "x"))},
            context=_CTX,
        ),
        PinningRow(
            case="dtype",
            feature="v__diff_1_offset",
            inputs={"p": (_S, ("a", "a")), "o": (_I, (1, 2)), "v": (DataType.UINT8, (5, 1))},
            # 1 - 5 = -4 does not wrap in uint8: diff of any integer source is int64
            expected={"v__diff_1_offset": (_I, (None, -4))},
            context=_CTX,
        ),
        PinningRow(
            case="dtype",
            feature="v__diff_1_offset",
            inputs={"p": (_S, ("a", "a")), "o": (_I, (1, 2)), "v": (DataType.INT8, (-100, 100))},
            # 100 - (-100) = 200 does not overflow int8
            expected={"v__diff_1_offset": (_I, (None, 200))},
            context=_CTX,
        ),
        PinningRow(
            case="dtype",
            feature="v__pct_change_1_offset",
            inputs={"p": (_S, ("a", "a")), "o": (_I, (1, 2)), "v": (_I, (2, 3))},
            expected={"v__pct_change_1_offset": (_D, (None, 0.5))},
            context=_CTX,
        ),
        PinningRow(
            case="empty",
            feature="v__lag_1_offset",
            inputs={"p": (_S, ()), "o": (_I, ()), "v": (_I, ())},
            expected={"v__lag_1_offset": (_I, ())},
            context=_CTX,
        ),
        PinningRow(
            case="empty",
            feature="v__first_value_offset",
            inputs={"p": (_S, ()), "o": (_I, ()), "v": (_D, ())},
            expected={"v__first_value_offset": (_D, ())},
            context=_CTX,
        ),
        PinningRow(
            case="empty",
            feature="v__pct_change_1_offset",
            inputs={"p": (_S, ()), "o": (_I, ()), "v": (_I, ())},
            expected={"v__pct_change_1_offset": (_D, ())},
            context=_CTX,
        ),
    ),
)
