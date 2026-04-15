"""Shared pandas helper utilities for group and window aggregation.

Centralizes the dropna=False and min_count=1 patterns so that every
pandas-based aggregation feature group handles null keys and all-null
groups consistently.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from mloda.community.feature_groups.data_operations.helper_columns import unique_helper_name

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
        _ddof: int = ddof

        def func(x: Any, _d: int = _ddof, _f: str = pandas_func) -> Any:
            return getattr(x, _f)(ddof=_d)

        return getattr(grouped, method)(func)

    kwargs: dict[str, Any] = {}
    if agg_type == "sum":
        kwargs["min_count"] = 1
    return getattr(grouped, method)(pandas_func, **kwargs)


def coerce_count_dtype(data: Any, feature_name: str, agg_type: str) -> None:
    """Cast *feature_name* column to int64 in-place when *agg_type* is ``"count"``."""
    if agg_type == "count":
        data[feature_name] = data[feature_name].astype("int64")


_MODE_IDX_COL = "__mloda_mode_row_idx__"
_MODE_COUNT_COL = "__mloda_mode_count__"
_MODE_FIRST_IDX_COL = "__mloda_mode_first_idx__"


def compute_mode_winners(
    data: pd.DataFrame,
    source_col: str,
    partition_by: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Return one row per partition with the mode value of *source_col*.

    Ties are broken by first occurrence in *data* (matching PyArrow
    ``mode_only``). Null values are ignored when counting (matching
    pandas ``Series.mode`` semantics); partitions whose source values
    are all-null are omitted from the result.

    The returned frame has columns ``partition_by + [source_col]`` and
    contains at most one row per unique partition-key combination.
    """
    partition_by = list(partition_by)
    work = data[partition_by + [source_col]].copy()
    idx_col = unique_helper_name(_MODE_IDX_COL, work.columns)
    count_col = unique_helper_name(_MODE_COUNT_COL, work.columns)
    first_idx_col = unique_helper_name(_MODE_FIRST_IDX_COL, work.columns)
    work[idx_col] = range(len(work))
    work = work[work[source_col].notna()]
    if work.empty:
        return data.iloc[0:0][partition_by + [source_col]].copy()

    counts = work.groupby(partition_by + [source_col], dropna=False, as_index=False).agg(
        **{
            count_col: (idx_col, "size"),
            first_idx_col: (idx_col, "min"),
        }
    )
    counts = counts.sort_values(
        partition_by + [count_col, first_idx_col],
        ascending=[True] * len(partition_by) + [False, True],
        kind="mergesort",
    )
    winners = counts.groupby(partition_by, dropna=False, as_index=False).head(1)
    return winners[partition_by + [source_col]].reset_index(drop=True)
