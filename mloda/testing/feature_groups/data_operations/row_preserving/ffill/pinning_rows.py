"""Hand-computed pinning rows of the ffill reference."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from mloda.user import DataType

from mloda.testing.feature_groups.data_operations.pinning import PinningRow, PinningTable

NAN = float("nan")

FFILL_RULES: Mapping[str, str] = MappingProxyType(
    {
        "order": "Each partition is ordered by order_by ascending; the last non-None value carries forward in that order.",
        "ties": "Rows tied on order_by keep input order.",
        "order_tiers": "In order_by real values come first, then NaN, then None; NaN and None are separate tiers.",
        "none": "Leading None values stay None; a None after a value gets the carried value; an all-None group stays None.",
        "nan": "NaN in the source is a value: it is carried forward like any other value.",
        "partition_keys": "NaN partition keys merge into one group; a None partition key is its own group.",
        "whole_table": "Without partition_by the whole table is one group.",
        "dtype": "The output keeps the source type.",
        "empty": "An empty input gives an empty column of the source type.",
    }
)

_I = DataType.INT64
_D = DataType.DOUBLE
_S = DataType.STRING
_CTX = {"partition_by": ["p"], "order_by": "o"}

FFILL_PINNING = PinningTable(
    operation="ffill",
    rules=FFILL_RULES,
    rows=(
        PinningRow(
            case="order",
            feature="v__ffill",
            inputs={"p": (_S, ("a",) * 3), "o": (_I, (3, 1, 2)), "v": (_I, (None, 5, None))},
            # order idx 1, 2, 0 holds 5, None, None
            expected={"v__ffill": (_I, (5, 5, 5))},
            context=_CTX,
        ),
        PinningRow(
            case="order",
            feature="v__ffill",
            inputs={"p": (_S, ("a",) * 4), "o": (_I, (4, 1, 3, 2)), "v": (_I, (9, None, 6, None))},
            # order idx 1, 3, 2, 0 holds None, None, 6, 9
            expected={"v__ffill": (_I, (9, None, 6, None))},
            context=_CTX,
        ),
        PinningRow(
            case="ties",
            feature="v__ffill",
            inputs={"p": (_S, ("a",) * 5), "o": (_I, (1,) * 5), "v": (_I, (None, 5, None, 7, None))},
            expected={"v__ffill": (_I, (None, 5, 5, 7, 7))},
            context=_CTX,
        ),
        PinningRow(
            case="order_tiers",
            feature="v__ffill",
            inputs={"p": (_S, ("a",) * 3), "o": (_D, (None, NAN, 1.0)), "v": (_I, (None, 20, None))},
            # order idx 2, 1, 0 holds None, 20, None
            expected={"v__ffill": (_I, (20, 20, None))},
            context=_CTX,
        ),
        PinningRow(
            case="order_tiers",
            feature="v__ffill",
            inputs={"p": (_S, ("a",) * 3), "o": (_I, (None, 1, 2)), "v": (_I, (5, None, None))},
            # order idx 1, 2, 0 holds None, None, 5: the None order value sorts last
            expected={"v__ffill": (_I, (5, None, None))},
            context=_CTX,
        ),
        PinningRow(
            case="none",
            feature="v__ffill",
            inputs={"p": (_S, ("a",) * 3), "o": (_I, (1, 2, 3)), "v": (_I, (None, 4, None))},
            expected={"v__ffill": (_I, (None, 4, 4))},
            context=_CTX,
        ),
        PinningRow(
            case="none",
            feature="v__ffill",
            inputs={"p": (_S, ("a", "a", "b")), "o": (_I, (1, 2, 1)), "v": (_I, (None, None, 3))},
            expected={"v__ffill": (_I, (None, None, 3))},
            context=_CTX,
        ),
        PinningRow(
            case="nan",
            feature="v__ffill",
            inputs={"p": (_S, ("a",) * 3), "o": (_I, (1, 2, 3)), "v": (_D, (1.0, NAN, None))},
            expected={"v__ffill": (_D, (1.0, NAN, NAN))},
            context=_CTX,
        ),
        PinningRow(
            case="nan",
            feature="v__ffill",
            inputs={"p": (_S, ("a",) * 3), "o": (_I, (1, 2, 3)), "v": (_D, (None, NAN, None))},
            expected={"v__ffill": (_D, (None, NAN, NAN))},
            context=_CTX,
        ),
        PinningRow(
            case="partition_keys",
            feature="v__ffill",
            inputs={"p": (_D, (NAN, 1.0, NAN)), "o": (_I, (1, 1, 2)), "v": (_I, (5, None, None))},
            expected={"v__ffill": (_I, (5, None, 5))},
            context=_CTX,
        ),
        PinningRow(
            case="partition_keys",
            feature="v__ffill",
            inputs={"p": (_D, (None, NAN, None, NAN)), "o": (_I, (1, 1, 2, 2)), "v": (_I, (1, None, None, None))},
            expected={"v__ffill": (_I, (1, None, 1, None))},
            context=_CTX,
        ),
        PinningRow(
            case="whole_table",
            feature="v__ffill",
            inputs={"o": (_I, (2, 1, 3)), "v": (_I, (1, None, None))},
            # order idx 1, 0, 2 holds None, 1, None
            expected={"v__ffill": (_I, (1, None, 1))},
            context={"order_by": "o"},
        ),
        PinningRow(
            case="dtype",
            feature="v__ffill",
            inputs={"p": (_S, ("a",) * 3), "o": (_I, (1, 2, 3)), "v": (_S, ("x", None, "y"))},
            expected={"v__ffill": (_S, ("x", "x", "y"))},
            context=_CTX,
        ),
        PinningRow(
            case="dtype",
            feature="v__ffill",
            inputs={"p": (_S, ("a",) * 2), "o": (_I, (1, 2)), "v": (DataType.INT32, (3, None))},
            expected={"v__ffill": (DataType.INT32, (3, 3))},
            context=_CTX,
        ),
        PinningRow(
            case="dtype",
            feature="v__ffill",
            inputs={"p": (_S, ("a",) * 2), "o": (_I, (1, 2)), "v": (DataType.BOOLEAN, (True, None))},
            expected={"v__ffill": (DataType.BOOLEAN, (True, True))},
            context=_CTX,
        ),
        PinningRow(
            case="empty",
            feature="v__ffill",
            inputs={"p": (_S, ()), "o": (_I, ()), "v": (_D, ())},
            expected={"v__ffill": (_D, ())},
            context=_CTX,
        ),
        PinningRow(
            case="none",
            feature="v__ffill",
            inputs={"p": (_S, ("a", "a")), "o": (_I, (1, 2)), "v": (_I, (None, None))},
            expected={"v__ffill": (_I, (None, None))},
            context=_CTX,
        ),
    ),
)
