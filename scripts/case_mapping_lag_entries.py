#!/usr/bin/env python3
"""Print override-table lines PyArrow's case mapping needs; run it with each interpreter whose drift test fails."""

from __future__ import annotations

import sys
from collections.abc import Callable, Collection, Sequence


def _render(text: str) -> str:
    """Return ``text`` as the body of a Python string literal, escaping supplementary and non-printable chars."""
    parts = []
    for ch in text:
        cp = ord(ch)
        if cp > 0xFFFF:
            parts.append(f"\\U{cp:08x}")
        elif not ch.isprintable() or ch in '"\\':
            parts.append(f"\\u{cp:04x}")
        else:
            parts.append(ch)
    return "".join(parts)


def _codepoints(text: str) -> str:
    return " ".join(f"U+{ord(ch):04X}" for ch in text)


def missing_entries(
    chars: Sequence[str], reference: Sequence[str], ours: Callable[[str], str], existing: Collection[str]
) -> list[str]:
    """Return one override-dict line per char whose ``ours`` mapping differs from ``reference``."""
    lines = []
    for ch, expected in zip(chars, reference):
        if ours(ch) == expected:
            continue
        suffix = " (replaces existing entry)" if ch in existing else ""
        lines.append(
            f'    "{_render(ch)}": "{_render(expected)}",  # {_codepoints(ch)} -> {_codepoints(expected)}{suffix}'
        )
    return lines


def main() -> int:
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
    except ImportError:
        print("pyarrow is required to compute the reference mappings.", file=sys.stderr)
        return 1
    from mloda.community.feature_groups.data_operations.string.python_dict_string import (
        _LOWER_OVERRIDES,
        _UPPER_OVERRIDES,
        _lower,
        _upper,
    )

    chars = [chr(cp) for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]
    array = pa.array(chars, type=pa.string())
    upper = missing_entries(chars, pc.utf8_upper(array).to_pylist(), _upper, _UPPER_OVERRIDES)
    lower = missing_entries(chars, pc.utf8_lower(array).to_pylist(), _lower, _LOWER_OVERRIDES)
    if not upper and not lower:
        print("No entries are missing on this interpreter.")
        return 0
    if upper:
        print("# _UPPER_OVERRIDES")
        print("\n".join(upper))
    if lower:
        print("# _LOWER_OVERRIDES")
        print("\n".join(lower))
    return 0


if __name__ == "__main__":
    sys.exit(main())
