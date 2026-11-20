"""Pure Python reference for string operations: runs the group's loop with a Python compute.

Reproduces Unicode simple case mapping (PyArrow utf8_upper/utf8_lower) from the interpreter's Unicode database.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mloda.provider import FeatureSet

if TYPE_CHECKING:
    import pyarrow as pa

STRING_OP_NAMES = frozenset({"upper", "lower", "trim", "length", "reverse"})


def _upper_char(ch: str) -> str:
    if ch == "ß":
        return "ẞ"
    for candidate in (ch.upper(), ch.title()):
        if len(candidate) == 1:
            return candidate
    return ch


def _lower_char(ch: str) -> str:
    if ch == "İ":
        return "i"
    lowered = ch.lower()
    return lowered if len(lowered) == 1 else ch


def _element(value: str, op: str) -> Any:
    if op == "upper":
        return "".join(_upper_char(ch) for ch in value)
    if op == "lower":
        return "".join(_lower_char(ch) for ch in value)
    if op == "trim":
        return value.strip()
    if op == "length":
        return len(value)
    return value[::-1]


def _compute(table: pa.Table, feature_name: str, source_col: str, op: str) -> pa.Table:
    import pyarrow as pa

    if op not in STRING_OP_NAMES:
        raise ValueError(f"Unsupported string operation {op!r}. Supported: {sorted(STRING_OP_NAMES)}")
    arrow_type = table.column(source_col).type
    if not (
        pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type) or pa.types.is_string_view(arrow_type)
    ):
        raise ValueError(f"String operation {op!r} requires a string source column, but {source_col!r} is {arrow_type}")
    values = [None if v is None else _element(v, op) for v in table.column(source_col).to_pylist()]
    declared = pa.int64() if op == "length" else pa.string()
    return table.append_column(feature_name, pa.array(values, type=declared))


def reference_string(group: Any, table: pa.Table, features: FeatureSet) -> pa.Table:
    """The table plus one pure Python column per string feature."""
    return group.transform_string(table, features, compute_string=_compute)
