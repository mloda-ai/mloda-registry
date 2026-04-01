"""PyArrow implementation for window aggregation feature groups.

Uses PyArrow's native ``Table.group_by().aggregate()`` API for vectorized,
C++-backed aggregation. The aggregate is computed per partition in C++ and
then broadcast back to every row via an index-list collected during the
same group_by call. Median and mode fall back to a list-collect-then-compute
path because PyArrow has no exact grouped median or grouped mode function.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional, Set, Type, Union

import pyarrow as pa
import pyarrow.compute as pc

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

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


class PyArrowWindowAggregation(WindowAggregationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PyArrowTable}

    @classmethod
    def _compute_window(
        cls,
        table: pa.Table,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
        order_by: Optional[str] = None,
    ) -> pa.Table:
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

        elif agg_type in ("median", "mode"):
            return cls._compute_via_list(t_with_idx, feature_name, source_col, partition_by, agg_type)

        else:
            raise ValueError(f"Unsupported aggregation type: {agg_type}")

        return cls._broadcast(table, grouped, agg_col, feature_name, num_rows)

    @classmethod
    def _compute_ordered(
        cls,
        t_with_idx: pa.Table,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
        order_by: Optional[str],
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
    def _compute_via_list(
        cls,
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
                agg_val = _median(non_null)
            else:
                agg_val = _mode(non_null)

            for idx in indices:
                result_values[idx] = agg_val

        original_table = t_with_idx.drop_columns([_IDX_COL])
        return original_table.append_column(feature_name, pa.array(result_values))

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


def _median(values: list[Any]) -> Any:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return float(s[mid])


def _mode(values: list[Any]) -> Any:
    if not values:
        return None
    counts = Counter(values)
    return counts.most_common(1)[0][0]
