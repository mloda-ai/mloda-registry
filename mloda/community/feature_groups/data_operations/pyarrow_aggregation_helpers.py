"""Shared PyArrow aggregation helpers for data operation feature groups.

PyArrow lacks native group-by / window-function APIs, so these packages use
Python dict-based grouping with row-by-row .as_py() calls. The helper
functions below compute scalar aggregates over plain Python lists and are
shared by pyarrow_group_aggregation and pyarrow_window_aggregation.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def aggregate(values: list[Any], agg_type: str) -> Any:
    """Compute a single aggregate over a list of values (may contain None)."""
    non_null = [v for v in values if v is not None]

    if not non_null:
        if agg_type in ("count", "nunique"):
            return 0
        return None

    if agg_type == "sum":
        return sum(non_null)
    if agg_type == "avg":
        return sum(non_null) / len(non_null)
    if agg_type == "count":
        return len(non_null)
    if agg_type == "min":
        return min(non_null)
    if agg_type == "max":
        return max(non_null)
    if agg_type == "std":
        return std(non_null)
    if agg_type == "var":
        return var(non_null)
    if agg_type == "median":
        return median(non_null)
    if agg_type == "mode":
        return mode(non_null)
    if agg_type == "nunique":
        return len(set(non_null))
    if agg_type == "first":
        return non_null[0]
    if agg_type == "last":
        return non_null[-1]

    raise ValueError(f"Unsupported aggregation type: {agg_type}")


def std(values: list[Any]) -> Any:
    if len(values) < 2:
        return None
    return var(values) ** 0.5


def var(values: list[Any]) -> Any:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / (len(values) - 1)


def median(values: list[Any]) -> Any:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return float(s[mid])


def mode(values: list[Any]) -> Any:
    if not values:
        return None
    counts = Counter(values)
    return counts.most_common(1)[0][0]
