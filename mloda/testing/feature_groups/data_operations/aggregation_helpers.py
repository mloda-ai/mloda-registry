"""Shared aggregation helpers for test reference implementations.

The helper functions below compute scalar aggregates over plain Python lists.
They are used by frame_aggregate (per-window aggregation) and as a fallback
for operations that PyArrow's native group_by API does not support directly.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error

# Sentinel key for NaN in mode's Counter: values come from to_pylist(), so every NaN
# is a distinct float object and Counter cannot merge them by equality.
_NAN_KEY = object()


def _is_nan(v: Any) -> bool:
    return isinstance(v, float) and math.isnan(v)


_SUPPORTED_AGG_TYPES = {
    "sum",
    "avg",
    "count",
    "min",
    "max",
    "std",
    "std_pop",
    "std_samp",
    "var",
    "var_pop",
    "var_samp",
    "median",
    "mode",
    "nunique",
    "first",
    "last",
}


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
    if agg_type in ("min", "max"):
        non_nan = [v for v in non_null if not _is_nan(v)]
        if not non_nan:
            # non_null was non-empty and entirely NaN: pc.min/pc.max skip NaN but an
            # all-NaN group still returns NaN (unlike median/quantile, which return null).
            return float("nan")
        return min(non_nan) if agg_type == "min" else max(non_nan)
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

    raise unsupported_agg_type_error(agg_type, _SUPPORTED_AGG_TYPES)


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
    """Median, skipping NaN like pc.quantile (all-NaN, like all-null, returns None)."""
    non_nan = [v for v in values if not _is_nan(v)]
    if not non_nan:
        return None
    s = sorted(non_nan)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return float(s[mid])


def mode(values: list[Any]) -> Any:
    """Mode, counting NaN as one value like pc.mode; ties keep first occurrence."""
    if not values:
        return None
    counts = Counter(_NAN_KEY if _is_nan(v) else v for v in values)
    winner = counts.most_common(1)[0][0]
    return float("nan") if winner is _NAN_KEY else winner
