"""Pure Python kernel shared by the aggregation, window, frame and scalar references.

Aggregates run over plain Python lists, independent of PyArrow compute and of the
production PythonDict helpers. The contract is in docs/guides/data-operation-patterns/03-reference-implementation.md.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import pyarrow as pa

from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error

# Sentinel key for NaN in mode, nunique and group keys: values come from to_pylist(), so every NaN
# is a distinct float object and equality cannot merge them.
_NAN_KEY = object()


def _is_nan(v: Any) -> bool:
    return isinstance(v, float) and math.isnan(v)


_SUPPORTED_AGG_TYPES = {
    "sum",
    "avg",
    "mean",
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
    if agg_type in ("avg", "mean"):
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
        return len({_NAN_KEY if _is_nan(v) else v for v in non_null})
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
    return sum((x - mean) * (x - mean) for x in values) / (len(values) - ddof)


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


def group_rows(key_columns: list[list[Any]], num_rows: int) -> dict[tuple[Any, ...], list[int]]:
    """Row indices per key tuple in first-occurrence order; NaN keys merge, None stays distinct, no keys gives ()."""
    groups: dict[tuple[Any, ...], list[int]] = {}
    for i in range(num_rows):
        key = tuple(_NAN_KEY if _is_nan(col[i]) else col[i] for col in key_columns)
        groups.setdefault(key, []).append(i)
    if not key_columns:
        groups.setdefault((), [])
    return groups


def order_key(value: Any) -> tuple[int, Any]:
    """Sort key: real values ascending, then NaN, then None."""
    if value is None:
        return (2, 0)
    if _is_nan(value):
        return (1, 0)
    return (0, value)


def declared_type(agg_type: str, source_type: pa.DataType) -> pa.DataType:
    """Output type of an aggregate column."""
    if agg_type in ("count", "nunique"):
        return pa.int64()
    if agg_type == "sum":
        if pa.types.is_signed_integer(source_type):
            return pa.int64()
        if pa.types.is_unsigned_integer(source_type):
            return pa.uint64()
        if pa.types.is_floating(source_type) or pa.types.is_null(source_type):
            return pa.float64()
        raise ValueError(f"Aggregation 'sum' is not defined for source type {source_type}")
    if agg_type in ("min", "max", "first", "last", "mode"):
        return source_type
    if agg_type in _SUPPORTED_AGG_TYPES:
        return pa.float64()
    raise unsupported_agg_type_error(agg_type, _SUPPORTED_AGG_TYPES)
