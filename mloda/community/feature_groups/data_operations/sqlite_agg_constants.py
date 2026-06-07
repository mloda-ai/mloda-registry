"""Shared SQLite aggregation-function table.

The aggregation, window-aggregation, and scalar-aggregate families map the
same aggregation names to the same native SQLite functions. (frame-aggregate
intentionally omits ``mean`` and keeps its own subset.)
"""

from __future__ import annotations

# Aggregation types that SQLite supports natively.
SQLITE_AGG_FUNCS: dict[str, str] = {
    "sum": "SUM",
    "avg": "AVG",
    "mean": "AVG",
    "count": "COUNT",
    "min": "MIN",
    "max": "MAX",
}
