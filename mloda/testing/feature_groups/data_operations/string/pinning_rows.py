"""Hand-computed pinning rows of the string reference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType

from mloda.user import DataType

from mloda.testing.feature_groups.data_operations.pinning import PinningRow, PinningTable

STRING_RULES: Mapping[str, str] = MappingProxyType(
    {
        "none": "None in gives None out, for every operation.",
        "empty": "An empty string stays empty (length 0); an empty input gives an empty typed column.",
        "case_mapping": (
            "upper and lower map each codepoint on its own (simple case mapping): no final sigma, no expansion."
        ),
        "trim": "trim strips Python str.strip() whitespace, including U+3000, U+0085 and U+001C, but not U+200B.",
        "length": "length counts codepoints, not user-perceived characters.",
        "reverse": "reverse reverses codepoints, so a combining mark moves before its base.",
        "dtype": "upper, lower, trim and reverse give string; length gives int64.",
        "row_order": "Output rows keep the input order.",
    }
)

_S = DataType.STRING
_I = DataType.INT64

_Pairs = Sequence[tuple[str | None, str | int | None]]


def _row(case: str, op: str, pairs: _Pairs, out: DataType = _S) -> PinningRow:
    feature = f"name__{op}"
    return PinningRow(
        case=case,
        feature=feature,
        inputs={"name": (_S, tuple(source for source, _ in pairs))},
        expected={feature: (out, tuple(value for _, value in pairs))},
    )


STRING_PINNING = PinningTable(
    operation="string",
    rules=STRING_RULES,
    rows=(
        _row("none", "upper", [("a", "A"), (None, None)]),
        _row("none", "lower", [("A", "a"), (None, None)]),
        _row("none", "trim", [(" a ", "a"), (None, None)]),
        _row("none", "length", [("ab", 2), (None, None)], _I),
        _row("none", "reverse", [("ab", "ba"), (None, None)]),
        _row("none", "upper", [(None, None), (None, None)]),
        _row("none", "length", [(None, None), (None, None)], _I),
        _row("empty", "upper", [("", ""), ("a", "A")]),
        _row("empty", "lower", [("", ""), ("A", "a")]),
        _row("empty", "trim", [("", ""), ("   ", "")]),
        _row("empty", "length", [("", 0), ("a", 1)], _I),
        _row("empty", "reverse", [("", ""), ("ab", "ba")]),
        _row("empty", "upper", []),
        _row("empty", "lower", []),
        _row("empty", "trim", []),
        _row("empty", "length", [], _I),
        _row("empty", "reverse", []),
        _row("case_mapping", "upper", [("ß", "ẞ"), ("ǅ", "Ǆ"), ("é", "É"), ("abc", "ABC")]),
        _row(
            "case_mapping",
            "lower",
            [("İ", "i"), ("ΟΔΟΣ", "οδοσ"), ("ǅ", "ǆ"), ("É", "é")],
        ),
        _row(
            "trim",
            "trim",
            [
                ("  a b \t\n", "a b"),
                ("\u3000x\u3000", "x"),
                ("\x85x\x1c", "x"),
                ("\u200bx\u200b", "\u200bx\u200b"),
                ("   ", ""),
            ],
        ),
        _row("length", "length", [("abc", 3), ("e\u0301", 2), ("\U0001f600", 1), ("", 0)], _I),
        _row("reverse", "reverse", [("e\u0301", "\u0301e"), ("\U0001f600x", "x\U0001f600"), ("abc", "cba")]),
        _row("dtype", "upper", [("a", "A")]),
        _row("dtype", "lower", [("A", "a")]),
        _row("dtype", "trim", [(" a", "a")]),
        _row("dtype", "length", [("abc", 3)], _I),
        _row("dtype", "reverse", [("ab", "ba")]),
        _row("row_order", "upper", [("b", "B"), ("a", "A"), ("c", "C")]),
        _row("row_order", "length", [("bbb", 3), ("a", 1), ("cc", 2)], _I),
    ),
)
