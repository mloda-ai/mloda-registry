"""Hand-computed pinning rows of the lookup reference."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from mloda.user import DataType

from mloda.testing.feature_groups.data_operations.pinning import PinningRow, PinningTable

LOOKUP_RULES: Mapping[str, str] = MappingProxyType(
    {
        "null_key": "A null source value stays null under every policy and is never counted as unmatched.",
        "unmatched": "An unmatched key gives null under null and passes the source value through under keep.",
        "key_type": (
            "String keys match string columns, int keys match integer columns by Python value: a key outside "
            "the column width never matches and 2**60 and 2**60 + 1 map exactly."
        ),
        "value_type": "The output type follows the mapping values: str STRING, int INT64, float DOUBLE, bool BOOLEAN.",
        "all_null": "A typed all-null column whose type matches the keys gives all null of the value type.",
        "row_order": "Output rows keep the input order; an empty input gives an empty column of the value type.",
    }
)

_S = DataType.STRING
_I = DataType.INT64
_D = DataType.DOUBLE
_B = DataType.BOOLEAN
_F = "a__lookup"


def _ctx(mapping: dict[Any, Any], on_unmatched: str) -> dict[str, Any]:
    return {"lookup_mapping": mapping, "on_unmatched": on_unmatched}


LOOKUP_PINNING = PinningTable(
    operation="lookup",
    rules=LOOKUP_RULES,
    rows=(
        PinningRow(
            case="null_key",
            feature=_F,
            inputs={"a": (_S, ("x", None, "y"))},
            expected={_F: (_S, ("X", None, "Y"))},
            context=_ctx({"x": "X", "y": "Y"}, "raise"),
        ),
        PinningRow(
            case="null_key",
            feature=_F,
            inputs={"a": (_S, ("x", None))},
            expected={_F: (_S, ("X", None))},
            context=_ctx({"x": "X"}, "null"),
        ),
        PinningRow(
            case="null_key",
            feature=_F,
            inputs={"a": (_S, ("x", None, "z"))},
            expected={_F: (_S, ("X", None, "z"))},
            context=_ctx({"x": "X"}, "keep"),
        ),
        PinningRow(
            case="unmatched",
            feature=_F,
            inputs={"a": (_S, ("x", "z", "x"))},
            expected={_F: (_S, ("X", None, "X"))},
            context=_ctx({"x": "X"}, "null"),
        ),
        PinningRow(
            case="unmatched",
            feature=_F,
            inputs={"a": (_I, (1, 3))},
            expected={_F: (_I, (10, 3))},
            context=_ctx({1: 10}, "keep"),
        ),
        PinningRow(
            case="key_type",
            feature=_F,
            inputs={"a": (_S, ("A100", "B200"))},
            expected={_F: (_S, ("Alpha", "Beta"))},
            context=_ctx({"A100": "Alpha", "B200": "Beta"}, "raise"),
        ),
        PinningRow(
            case="key_type",
            feature=_F,
            inputs={"a": (DataType.INT8, (1, -5, 7))},
            expected={_F: (_I, (10, 50, None))},
            context=_ctx({1: 10, -5: 50}, "null"),
        ),
        PinningRow(
            case="key_type",
            feature=_F,
            inputs={"a": (DataType.INT32, (1, 5, None))},
            expected={_F: (_I, (10, None, None))},
            context=_ctx({1: 10, 2**40: 20}, "null"),
        ),
        PinningRow(
            case="key_type",
            feature=_F,
            inputs={"a": (DataType.UINT8, (1, 5))},
            expected={_F: (_I, (10, 5))},
            context=_ctx({1: 10, -1: 20}, "keep"),
        ),
        PinningRow(
            case="key_type",
            feature=_F,
            inputs={"a": (_I, (2**60, 2**60 + 1, 7))},
            expected={_F: (_S, ("lo", "hi", None))},
            context=_ctx({2**60: "lo", 2**60 + 1: "hi"}, "null"),
        ),
        PinningRow(
            case="key_type",
            feature=_F,
            inputs={"a": (_I, (2**60, 2**60 + 1))},
            expected={_F: (_I, (1, 2))},
            context=_ctx({2**60: 1, 2**60 + 1: 2}, "raise"),
        ),
        PinningRow(
            case="value_type",
            feature=_F,
            inputs={"a": (_S, ("x", "y"))},
            expected={_F: (_S, ("X", "Y"))},
            context=_ctx({"x": "X", "y": "Y"}, "raise"),
        ),
        PinningRow(
            case="value_type",
            feature=_F,
            inputs={"a": (_S, ("x", "z"))},
            expected={_F: (_I, (1, None))},
            context=_ctx({"x": 1}, "null"),
        ),
        PinningRow(
            case="value_type",
            feature=_F,
            inputs={"a": (_S, ("x", "y"))},
            expected={_F: (_D, (1.5, 2.5))},
            context=_ctx({"x": 1.5, "y": 2.5}, "raise"),
        ),
        PinningRow(
            case="value_type",
            feature=_F,
            inputs={"a": (_S, ("x", "y", "z"))},
            expected={_F: (_B, (True, False, None))},
            context=_ctx({"x": True, "y": False}, "null"),
        ),
        PinningRow(
            case="value_type",
            feature=_F,
            inputs={"a": (DataType.INT32, (1, 3))},
            expected={_F: (_I, (10, 3))},
            context=_ctx({1: 10}, "keep"),
        ),
        PinningRow(
            case="all_null",
            feature=_F,
            inputs={"a": (_I, (None, None))},
            expected={_F: (_S, (None, None))},
            context=_ctx({1: "a"}, "raise"),
        ),
        PinningRow(
            case="all_null",
            feature=_F,
            inputs={"a": (_S, (None,))},
            expected={_F: (_D, (None,))},
            context=_ctx({"x": 1.5}, "null"),
        ),
        PinningRow(
            case="all_null",
            feature=_F,
            inputs={"a": (_I, (None, None))},
            expected={_F: (_I, (None, None))},
            context=_ctx({1: 10}, "keep"),
        ),
        PinningRow(
            case="row_order",
            feature=_F,
            inputs={"a": (_S, ("b", "a", "b"))},
            expected={_F: (_S, ("B", "A", "B"))},
            context=_ctx({"a": "A", "b": "B"}, "raise"),
        ),
        PinningRow(
            case="row_order",
            feature=_F,
            inputs={"a": (_S, ())},
            expected={_F: (_I, ())},
            context=_ctx({"a": 1}, "raise"),
        ),
    ),
)
