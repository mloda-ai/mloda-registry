"""Pinning rows of a reference and their markdown rendering for the guides."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from mloda.user import DataType

PINNING_BEGIN_MARKER = "<!-- BEGIN GENERATED: reference-pinning -->"
PINNING_END_MARKER = "<!-- END GENERATED: reference-pinning -->"

TypedColumns = Mapping[str, tuple[DataType, tuple[Any, ...]]]


@dataclass(frozen=True)
class PinningRow:
    """One hand-computed case: typed input columns, the feature, its context and typed expected columns."""

    case: str
    feature: str
    inputs: TypedColumns
    expected: TypedColumns
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PinningTable:
    """The rows of one operation plus the rules (label to text) its rows pin."""

    operation: str
    rules: Mapping[str, str]
    rows: tuple[PinningRow, ...]


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _value(value: Any) -> str:
    if isinstance(value, float) and value != value:
        return "NaN"
    if isinstance(value, (float, str)):
        return repr(value)
    return str(value)


def _context_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_value(item) for item in value) + "]"
    return _value(value)


def _columns(columns: TypedColumns) -> str:
    return "; ".join(
        f"{name}=[{', '.join(_value(v) for v in values)}] ({data_type.value})"
        for name, (data_type, values) in columns.items()
    )


def render_pinning_block(tables: Sequence[PinningTable]) -> str:
    """The generated block: rule tables grouped by identical rules, then the rows of each table in the order given."""
    shown = [table for table in tables if table.rows]
    out: list[str] = [PINNING_BEGIN_MARKER, ""]
    out.append("Rendered from each reference's `pinning_rows.py`; every row is run against its reference.")
    out.append("")
    groups: list[list[PinningTable]] = []
    for table in shown:
        for group in groups:
            if dict(group[0].rules) == dict(table.rules):
                group.append(table)
                break
        else:
            groups.append([table])
    for group in groups:
        out.append(f"### Rules ({', '.join(t.operation for t in group)})")
        out.append("")
        out.append("| Case | Rule | Pinned in |")
        out.append("|---|---|---|")
        for label, rule in group[0].rules.items():
            users = ", ".join(t.operation for t in group if any(row.case == label for row in t.rows))
            out.append(f"| {_cell(label)} | {_cell(rule)} | {users} |")
        out.append("")
    for table in shown:
        out.append(f"### {table.operation}")
        out.append("")
        out.append("| Case | Feature | Context | Input | Expected |")
        out.append("|---|---|---|---|---|")
        for row in table.rows:
            context = ", ".join(f"{k}={_context_value(v)}" for k, v in row.context.items()) or "-"
            cells = [row.case, f"`{row.feature}`", context, _columns(row.inputs), _columns(row.expected)]
            out.append("| " + " | ".join(_cell(c) for c in cells) + " |")
        out.append("")
    out.append(PINNING_END_MARKER)
    return "\n".join(out) + "\n"
