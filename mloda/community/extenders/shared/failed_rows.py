"""Counts and reason codes of failed rows, never positions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypedDict

from mloda.provider import FailedRows


class ColumnSummary(TypedDict):
    count: int
    reasons: list[str]


class FailedRowsSummary(TypedDict):
    count: int
    columns: dict[str, ColumnSummary]


def failed_rows_summary(entries: Iterable[FailedRows]) -> FailedRowsSummary:
    """Distinct failed-row counts overall and per column, with sorted reason codes."""
    by_column: dict[str, tuple[set[int], set[str]]] = {}
    for entry in entries:
        positions, reasons = by_column.setdefault(entry.column, (set(), set()))
        positions.update(entry.rows)
        reasons.add(entry.reason)
    total = len(set().union(*(p for p, _ in by_column.values())))
    columns: dict[str, ColumnSummary] = {
        column: {"count": len(by_column[column][0]), "reasons": sorted(by_column[column][1])}
        for column in sorted(by_column)
    }
    return {"count": total, "columns": columns}
