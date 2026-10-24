"""Test reference for window aggregation: the shared loop with a PyArrow-backed compute.

Accepts PyArrow tables but computes in Python for median and mode. Used as
the cross-framework comparison baseline in test suites. Native PyArrow
group_by aggregation is used for sum, avg, count, min, max, variance, stddev,
nunique, first, and last. Median and mode fall back to a list-collect-then-
compute path because PyArrow has no exact grouped median or grouped mode
function.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from mloda.provider import FeatureSet

from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error
from mloda.community.feature_groups.data_operations.mask_utils import apply_pyarrow_mask
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
    WindowAggregationFeatureGroup,
)
from mloda.testing.feature_groups.data_operations import aggregation_helpers

_IDX_COL = "__mloda_wa_idx__"
_MASKED_COL = "__mloda_wa_masked__"

_PA_AGG_FUNCS: dict[str, str] = {
    "sum": "sum",
    "avg": "mean",
    "mean": "mean",
    "count": "count",
    "min": "min",
    "max": "max",
    "nunique": "count_distinct",
}

_VARIANCE_FUNCS: dict[str, tuple[str, int]] = {
    "std": ("stddev", 0),
    "var": ("variance", 0),
    "std_pop": ("stddev", 0),
    "std_samp": ("stddev", 1),
    "var_pop": ("variance", 0),
    "var_samp": ("variance", 1),
}

_ORDERED_FUNCS: dict[str, str] = {
    "first": "first",
    "last": "last",
}

_SUPPORTED_AGG_TYPES = {*_PA_AGG_FUNCS, *_VARIANCE_FUNCS, *_ORDERED_FUNCS, "median", "mode"}


def _compute_window(
    table: pa.Table,
    feature_name: str,
    source_col: str,
    partition_by: list[str],
    agg_type: str,
    order_by: str | None = None,
    mask_spec: list[tuple[str, str, Any]] | None = None,
) -> pa.Table:
    num_rows = table.num_rows
    t_with_idx = table.append_column(_IDX_COL, pa.array(range(num_rows)))

    # Keys and order stay unmasked; the masked values are aggregated under a temp name.
    agg_source = source_col
    if mask_spec is not None:
        agg_source = _MASKED_COL
        t_with_idx = t_with_idx.append_column(_MASKED_COL, apply_pyarrow_mask(table, source_col, mask_spec))

    if agg_type in _PA_AGG_FUNCS:
        pa_func = _PA_AGG_FUNCS[agg_type]
        grouped = t_with_idx.group_by(partition_by).aggregate(
            [
                (agg_source, pa_func),
                (_IDX_COL, "list"),
            ]
        )
        agg_col = f"{agg_source}_{pa_func}"

    elif agg_type in _VARIANCE_FUNCS:
        pa_func, ddof = _VARIANCE_FUNCS[agg_type]
        grouped = t_with_idx.group_by(partition_by).aggregate(
            [
                (agg_source, pa_func, pc.VarianceOptions(ddof=ddof)),
                (_IDX_COL, "list"),
            ]
        )
        agg_col = f"{agg_source}_{pa_func}"

    elif agg_type in _ORDERED_FUNCS:
        return _drop_masked(_compute_ordered(t_with_idx, feature_name, agg_source, partition_by, agg_type, order_by))

    elif agg_type in ("median", "mode"):
        return _drop_masked(_compute_via_list(t_with_idx, feature_name, agg_source, partition_by, agg_type))

    else:
        raise unsupported_agg_type_error(agg_type, _SUPPORTED_AGG_TYPES, framework="Reference")

    return _broadcast(table, grouped, agg_col, feature_name, num_rows)


def _drop_masked(table: pa.Table) -> pa.Table:
    return table.drop_columns([_MASKED_COL]) if _MASKED_COL in table.column_names else table


def _compute_ordered(
    t_with_idx: pa.Table,
    feature_name: str,
    source_col: str,
    partition_by: list[str],
    agg_type: str,
    order_by: str | None,
) -> pa.Table:
    pa_func = _ORDERED_FUNCS[agg_type]
    sort_keys = [(col, "ascending") for col in partition_by]
    if order_by:
        sort_keys.append((order_by, "ascending"))
    indices = pc.sort_indices(t_with_idx, sort_keys=sort_keys)
    sorted_t = t_with_idx.take(indices)

    grouped = sorted_t.group_by(partition_by, use_threads=False).aggregate(
        [
            (source_col, pa_func),
            (_IDX_COL, "list"),
        ]
    )
    agg_col = f"{source_col}_{pa_func}"

    original_table = t_with_idx.drop_columns([_IDX_COL])
    return _broadcast(original_table, grouped, agg_col, feature_name, original_table.num_rows)


def _compute_via_list(
    t_with_idx: pa.Table,
    feature_name: str,
    source_col: str,
    partition_by: list[str],
    agg_type: str,
) -> pa.Table:
    grouped = t_with_idx.group_by(partition_by).aggregate(
        [
            (source_col, "list"),
            (_IDX_COL, "list"),
        ]
    )
    list_col = f"{source_col}_list"
    idx_list_col = f"{_IDX_COL}_list"

    num_rows = t_with_idx.num_rows
    result_values: list[Any] = [None] * num_rows

    for g in range(grouped.num_rows):
        vals = grouped.column(list_col)[g].as_py()
        indices = grouped.column(idx_list_col)[g].as_py()
        non_null = [v for v in vals if v is not None]

        if not non_null:
            agg_val = None
        elif agg_type == "median":
            agg_val = aggregation_helpers.median(non_null)
        else:
            agg_val = aggregation_helpers.mode(non_null)

        for idx in indices:
            result_values[idx] = agg_val

    original_table = t_with_idx.drop_columns([_IDX_COL])
    return original_table.append_column(feature_name, pa.array(result_values))


def _broadcast(
    original_table: pa.Table,
    grouped: pa.Table,
    agg_col: str,
    feature_name: str,
    num_rows: int,
) -> pa.Table:
    idx_list_col = f"{_IDX_COL}_list"
    result_values: list[Any] = [None] * num_rows

    for g in range(grouped.num_rows):
        agg_val = grouped.column(agg_col)[g].as_py()
        indices = grouped.column(idx_list_col)[g].as_py()
        for idx in indices:
            result_values[idx] = agg_val

    return original_table.append_column(feature_name, pa.array(result_values))


def reference_window_aggregation(table: pa.Table, features: FeatureSet) -> pa.Table:
    """Run the shared window aggregation loop with the PyArrow-backed reference compute."""
    return WindowAggregationFeatureGroup.aggregate(table, features, compute_window=_compute_window)
