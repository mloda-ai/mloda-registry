"""Shared PyArrow aggregation helpers for data operation feature groups.

The helper functions below compute scalar aggregates over plain Python lists.
They are used by frame_aggregate (per-window aggregation) and as a fallback
for operations that PyArrow's native group_by API does not support directly.
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
    if agg_type in ("std", "std_pop"):
        return std(non_null, ddof=0)
    if agg_type in ("var", "var_pop"):
        return var(non_null, ddof=0)
    if agg_type == "std_samp":
        return std(non_null, ddof=1)
    if agg_type == "var_samp":
        return var(non_null, ddof=1)
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


def std(values: list[Any], ddof: int = 0) -> Any:
    if len(values) < ddof + 1:
        return None
    return var(values, ddof=ddof) ** 0.5


def var(values: list[Any], ddof: int = 0) -> Any:
    if len(values) < ddof + 1:
        return None
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / (len(values) - ddof)


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
