"""Unit tests for scripts/case_mapping_lag_entries.py: missing_entries renders override-dict lines for lagging chars."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.script_loader import load_script

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"

lag = load_script("case_mapping_lag_entries", _SCRIPTS / "case_mapping_lag_entries.py")


def _identity(c: str) -> str:
    return c


def test_bmp_printable_char_is_rendered_as_literal_glyph() -> None:
    assert lag.missing_entries(["a"], ["Z"], str.upper, set()) == ['    "a": "Z",  # U+0061 -> U+005A']


def test_supplementary_char_is_rendered_as_escape() -> None:
    result = lag.missing_entries(["\U00016ebb"], ["\U00016ea0"], _identity, set())
    assert result == ['    "\\U00016ebb": "\\U00016ea0",  # U+16EBB -> U+16EA0']


def test_non_printable_bmp_char_is_rendered_as_escape() -> None:
    result = lag.missing_entries(["\u0378"], ["\u0379"], _identity, set())
    assert result == ['    "\\u0378": "\\u0379",  # U+0378 -> U+0379']


def test_agreeing_chars_emit_nothing() -> None:
    assert lag.missing_entries(["a", "b"], ["A", "B"], str.upper, set()) == []


def test_existing_key_is_marked_as_replaced() -> None:
    result = lag.missing_entries(["a"], ["Z"], str.upper, {"a"})
    assert result == ['    "a": "Z",  # U+0061 -> U+005A (replaces existing entry)']


@pytest.mark.parametrize(("key", "value"), [("\U00016ebb", "\U00016ea0"), ("\u0378", "\u0379")])
def test_rendered_line_round_trips_through_literal_eval(key: str, value: str) -> None:
    (line,) = lag.missing_entries([key], [value], _identity, set())
    assert ast.literal_eval("{\n" + line + "\n}") == {key: value}
