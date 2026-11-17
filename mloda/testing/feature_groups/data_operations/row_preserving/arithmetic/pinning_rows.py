"""Hand-computed pinning rows of the point and scalar arithmetic references."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from mloda.user import DataType

from mloda.testing.feature_groups.data_operations.pinning import PinningRow, PinningTable

NAN = float("nan")
INF = float("inf")

ARITHMETIC_RULES: Mapping[str, str] = MappingProxyType(
    {
        "none": "None in either operand gives None, also when the other operand is NaN.",
        "nan": "NaN is a value: it propagates through add, subtract, multiply and divide.",
        "divide_by_zero": (
            "Divide follows IEEE 754: x / 0 is inf with sign(a) * sign(b), 0 / 0 and NaN / 0 are NaN; "
            "the scalar group rejects a zero constant before compute."
        ),
        "signed_zero": "A zero result keeps its IEEE sign, so 0.0 * -1.0 is -0.0.",
        "dtype": (
            "Integer operands of any width, signed or unsigned, give int64; float64 when either operand is a float "
            "or the op is divide; an integer result outside its operands' integer range raises instead of wrapping; "
            "bool, string, decimal and null-typed sources are rejected."
        ),
        "row_order": "Output rows keep the input order, also for an empty input.",
    }
)

_I = DataType.INT64
_D = DataType.DOUBLE

POINT_ARITHMETIC_PINNING = PinningTable(
    operation="point_arithmetic",
    rules=ARITHMETIC_RULES,
    rows=(
        PinningRow(
            case="none",
            feature="a&b__add_point",
            inputs={"a": (_I, (1, None, 3)), "b": (_I, (None, 2, 4))},
            expected={"a&b__add_point": (_I, (None, None, 7))},
        ),
        PinningRow(
            case="none",
            feature="a&b__divide_point",
            inputs={"a": (_D, (None, 1.0)), "b": (_D, (2.0, None))},
            expected={"a&b__divide_point": (_D, (None, None))},
        ),
        PinningRow(
            case="nan",
            feature="a&b__multiply_point",
            inputs={"a": (_D, (NAN, 2.0, NAN)), "b": (_D, (1.0, NAN, None))},
            expected={"a&b__multiply_point": (_D, (NAN, NAN, None))},
        ),
        PinningRow(
            case="nan",
            feature="a&b__add_point",
            inputs={"a": (_D, (NAN, 1.0)), "b": (_D, (1.0, NAN))},
            expected={"a&b__add_point": (_D, (NAN, NAN))},
        ),
        PinningRow(
            case="nan",
            feature="a&b__subtract_point",
            inputs={"a": (_D, (1.0, NAN)), "b": (_D, (NAN, 1.0))},
            expected={"a&b__subtract_point": (_D, (NAN, NAN))},
        ),
        PinningRow(
            case="nan",
            feature="a&b__divide_point",
            inputs={"a": (_D, (NAN, 2.0)), "b": (_D, (2.0, NAN))},
            expected={"a&b__divide_point": (_D, (NAN, NAN))},
        ),
        PinningRow(
            case="divide_by_zero",
            feature="a&b__divide_point",
            inputs={
                "a": (_D, (1.0, -1.0, 1.0, -1.0, 0.0, NAN, 0.0)),
                "b": (_D, (0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0)),
            },
            expected={"a&b__divide_point": (_D, (INF, -INF, -INF, INF, NAN, NAN, NAN))},
        ),
        PinningRow(
            case="divide_by_zero",
            feature="a&b__divide_point",
            inputs={"a": (_I, (1, -1, 0)), "b": (_I, (0, 0, 0))},
            expected={"a&b__divide_point": (_D, (INF, -INF, NAN))},
        ),
        PinningRow(
            case="signed_zero",
            feature="a&b__multiply_point",
            inputs={"a": (_D, (0.0, -0.0, -0.0)), "b": (_D, (-1.0, -1.0, 1.0))},
            expected={"a&b__multiply_point": (_D, (-0.0, 0.0, -0.0))},
        ),
        PinningRow(
            case="signed_zero",
            feature="a&b__add_point",
            inputs={"a": (_D, (-0.0, -0.0, 0.0)), "b": (_D, (-0.0, 0.0, -0.0))},
            expected={"a&b__add_point": (_D, (-0.0, 0.0, 0.0))},
        ),
        PinningRow(
            case="signed_zero",
            feature="a&b__subtract_point",
            inputs={"a": (_D, (0.0, -0.0, 0.0)), "b": (_D, (0.0, 0.0, -0.0))},
            expected={"a&b__subtract_point": (_D, (0.0, -0.0, 0.0))},
        ),
        PinningRow(
            case="signed_zero",
            feature="a&b__divide_point",
            inputs={"a": (_D, (0.0, -0.0)), "b": (_D, (-1.0, -1.0))},
            expected={"a&b__divide_point": (_D, (-0.0, 0.0))},
        ),
        PinningRow(
            case="dtype",
            feature="a&b__add_point",
            inputs={"a": (DataType.INT32, (2147483646,)), "b": (DataType.INT32, (1,))},
            expected={"a&b__add_point": (_I, (2147483647,))},
        ),
        PinningRow(
            case="dtype",
            feature="a&b__multiply_point",
            inputs={"a": (DataType.INT8, (12, -12)), "b": (DataType.INT8, (10, 10))},
            expected={"a&b__multiply_point": (_I, (120, -120))},
        ),
        PinningRow(
            case="dtype",
            feature="a&b__subtract_point",
            inputs={"a": (DataType.UINT8, (3,)), "b": (DataType.UINT8, (2,))},
            expected={"a&b__subtract_point": (_I, (1,))},
        ),
        PinningRow(
            case="dtype",
            feature="a&b__add_point",
            inputs={"a": (DataType.UINT64, (5,)), "b": (DataType.UINT64, (7,))},
            expected={"a&b__add_point": (_I, (12,))},
        ),
        PinningRow(
            case="dtype",
            feature="a&b__add_point",
            inputs={"a": (_I, (1, 2)), "b": (_D, (0.5, 0.25))},
            expected={"a&b__add_point": (_D, (1.5, 2.25))},
        ),
        PinningRow(
            case="dtype",
            feature="a&b__add_point",
            inputs={"a": (DataType.INT32, (1,)), "b": (DataType.FLOAT, (0.5,))},
            expected={"a&b__add_point": (_D, (1.5,))},
        ),
        PinningRow(
            case="dtype",
            feature="a&b__divide_point",
            inputs={"a": (_I, (7, -7)), "b": (_I, (2, 2))},
            expected={"a&b__divide_point": (_D, (3.5, -3.5))},
        ),
        PinningRow(
            case="row_order",
            feature="a&b__add_point",
            inputs={"a": (_I, (3, 1, 2)), "b": (_I, (30, 10, 20))},
            expected={"a&b__add_point": (_I, (33, 11, 22))},
        ),
        PinningRow(
            case="row_order",
            feature="a&b__add_point",
            inputs={"a": (_I, ()), "b": (_I, ())},
            expected={"a&b__add_point": (_I, ())},
        ),
    ),
)

SCALAR_ARITHMETIC_PINNING = PinningTable(
    operation="scalar_arithmetic",
    rules=ARITHMETIC_RULES,
    rows=(
        PinningRow(
            case="none",
            feature="a__add_constant",
            inputs={"a": (_I, (1, None, 3))},
            expected={"a__add_constant": (_I, (2, None, 4))},
            context={"constant": 1},
        ),
        PinningRow(
            case="nan",
            feature="a__multiply_constant",
            inputs={"a": (_D, (NAN, None, 1.5))},
            expected={"a__multiply_constant": (_D, (NAN, None, 3.0))},
            context={"constant": 2.0},
        ),
        PinningRow(
            case="nan",
            feature="a__add_constant",
            inputs={"a": (_D, (NAN, 1.0))},
            expected={"a__add_constant": (_D, (NAN, 3.0))},
            context={"constant": 2.0},
        ),
        PinningRow(
            case="signed_zero",
            feature="a__multiply_constant",
            inputs={"a": (_D, (1.0, -1.0, 0.0, -0.0))},
            expected={"a__multiply_constant": (_D, (-0.0, 0.0, -0.0, 0.0))},
            context={"constant": -0.0},
        ),
        PinningRow(
            case="signed_zero",
            feature="a__multiply_constant",
            inputs={"a": (_D, (0.0, -0.0, 2.0))},
            expected={"a__multiply_constant": (_D, (-0.0, 0.0, -2.0))},
            context={"constant": -1},
        ),
        PinningRow(
            case="dtype",
            feature="a__add_constant",
            inputs={"a": (_I, (1, 2))},
            expected={"a__add_constant": (_D, (1.5, 2.5))},
            context={"constant": 0.5},
        ),
        PinningRow(
            case="dtype",
            feature="a__add_constant",
            inputs={"a": (DataType.INT32, (2147483646,))},
            expected={"a__add_constant": (_I, (2147483647,))},
            context={"constant": 1},
        ),
        PinningRow(
            case="dtype",
            feature="a__multiply_constant",
            inputs={"a": (_D, (1.0, 2.0))},
            expected={"a__multiply_constant": (_D, (3.0, 6.0))},
            context={"constant": 3},
        ),
        PinningRow(
            case="dtype",
            feature="a__divide_constant",
            inputs={"a": (_I, (7, -7))},
            expected={"a__divide_constant": (_D, (3.5, -3.5))},
            context={"constant": 2},
        ),
        PinningRow(
            case="dtype",
            feature="a__subtract_constant",
            inputs={"a": (DataType.UINT8, (3,))},
            expected={"a__subtract_constant": (_I, (1,))},
            context={"constant": 2},
        ),
        PinningRow(
            case="row_order",
            feature="a__subtract_constant",
            inputs={"a": (_I, (3, 1, 2))},
            expected={"a__subtract_constant": (_I, (2, 0, 1))},
            context={"constant": 1},
        ),
        PinningRow(
            case="row_order",
            feature="a__multiply_constant",
            inputs={"a": (_I, ())},
            expected={"a__multiply_constant": (_I, ())},
            context={"constant": 2},
        ),
    ),
)
