"""PyArrow implementation for window aggregation feature groups.

Uses PyArrow's native ``Table.group_by().aggregate()`` API for vectorized,
C++-backed aggregation. The aggregate is computed per partition in C++ and
then broadcast back to every row via an index-list collected during the
same group_by call.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error
from mloda.community.feature_groups.data_operations.mask_utils import apply_pyarrow_mask
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
    WindowAggregationFeatureGroup,
)

_IDX_COL = "__mloda_wa_idx__"

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

_SUPPORTED_AGG_TYPES = {*_PA_AGG_FUNCS, *_VARIANCE_FUNCS, *_ORDERED_FUNCS}


class PyArrowWindowAggregation(WindowAggregationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PyArrowTable}

    @classmethod
    def _compute_window(
        cls,
        table: pa.Table,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
        order_by: str | None = None,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> pa.Table:
        if mask_spec is not None:
            table = apply_pyarrow_mask(table, source_col, mask_spec)

        num_rows = table.num_rows
        t_with_idx = table.append_column(_IDX_COL, pa.array(range(num_rows)))

        if agg_type in _PA_AGG_FUNCS:
            pa_func = _PA_AGG_FUNCS[agg_type]
            grouped = t_with_idx.group_by(partition_by).aggregate(
                [
                    (source_col, pa_func),
                    (_IDX_COL, "list"),
                ]
            )
            agg_col = f"{source_col}_{pa_func}"

        elif agg_type in _VARIANCE_FUNCS:
            pa_func, ddof = _VARIANCE_FUNCS[agg_type]
            grouped = t_with_idx.group_by(partition_by).aggregate(
                [
                    (source_col, pa_func, pc.VarianceOptions(ddof=ddof)),
                    (_IDX_COL, "list"),
                ]
            )
            agg_col = f"{source_col}_{pa_func}"

        elif agg_type in _ORDERED_FUNCS:
            return cls._compute_ordered(t_with_idx, feature_name, source_col, partition_by, agg_type, order_by)

        else:
            raise unsupported_agg_type_error(agg_type, _SUPPORTED_AGG_TYPES, framework="PyArrow")

        return cls._broadcast(table, grouped, agg_col, feature_name, num_rows)

    @classmethod
    def _compute_ordered(
        cls,
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
        indices = pc.sort_indices(t_with_idx, sort_keys=sort_keys, null_placement="at_end")
        sorted_t = t_with_idx.take(indices)

        grouped = sorted_t.group_by(partition_by, use_threads=False).aggregate(
            [
                (source_col, pa_func),
                (_IDX_COL, "list"),
            ]
        )
        agg_col = f"{source_col}_{pa_func}"

        original_table = t_with_idx.drop_columns([_IDX_COL])
        return cls._broadcast(original_table, grouped, agg_col, feature_name, original_table.num_rows)

    @classmethod
    def _broadcast(
        cls,
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
