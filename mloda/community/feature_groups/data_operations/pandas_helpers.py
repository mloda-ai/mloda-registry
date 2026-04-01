"""Shared pandas helper utilities for group and window aggregation.

Centralizes the dropna=False and min_count=1 patterns so that every
pandas-based aggregation feature group handles null keys and all-null
groups consistently.
"""

from __future__ import annotations

from typing import Any

PANDAS_AGG_FUNCS: dict[str, str] = {
    "sum": "sum",
    "avg": "mean",
    "mean": "mean",
    "count": "count",
    "min": "min",
    "max": "max",
    "std": "std",
    "var": "var",
    "std_pop": "std",
    "std_samp": "std",
    "var_pop": "var",
    "var_samp": "var",
    "nunique": "nunique",
    "first": "first",
    "last": "last",
    "median": "median",
}


def null_safe_groupby(df: Any, partition_by: list[str], col: str) -> Any:
    """Group *df* by *partition_by* keeping null keys, then select *col*.

    Returns ``df.groupby(partition_by, dropna=False)[col]``.  When
    *partition_by* contains a single column the raw string is passed so
    that group keys are scalars rather than single-element tuples.
    """
    by: str | list[str] = partition_by[0] if len(partition_by) == 1 else partition_by
    return df.groupby(by, dropna=False)[col]


# Pandas groupby.agg("std") / .transform("std") always uses ddof=1.
# There is no way to pass ddof through the string-based API, so we
# intercept std/var operations and use a lambda wrapper instead.
_DDOF_BY_AGG_TYPE: dict[str, int] = {
    "std": 0,
    "var": 0,
    "std_pop": 0,
    "var_pop": 0,
    "std_samp": 1,
    "var_samp": 1,
}


def apply_null_safe_agg(
    grouped: Any,
    pandas_func: str,
    agg_type: str,
    *,
    method: str = "agg",
) -> Any:
    """Apply *pandas_func* via *method* on a grouped series.

    When *agg_type* is ``"sum"``, ``min_count=1`` is forwarded so that
    all-NaN groups return NaN rather than 0.

    When *agg_type* is a std/var variant, a lambda wrapper is used to
    pass the correct ``ddof`` (0 for population, 1 for sample) because
    the string-based pandas API does not support a ddof parameter.
    """
    ddof = _DDOF_BY_AGG_TYPE.get(agg_type)
    if ddof is not None:
        func = lambda x, _d=ddof, _f=pandas_func: getattr(x, _f)(ddof=_d)
        return getattr(grouped, method)(func)

    kwargs: dict[str, Any] = {}
    if agg_type == "sum":
        kwargs["min_count"] = 1
    return getattr(grouped, method)(pandas_func, **kwargs)


def coerce_count_dtype(data: Any, feature_name: str, agg_type: str) -> None:
    """Cast *feature_name* column to int64 in-place when *agg_type* is ``"count"``."""
    if agg_type == "count":
        data[feature_name] = data[feature_name].astype("int64")
