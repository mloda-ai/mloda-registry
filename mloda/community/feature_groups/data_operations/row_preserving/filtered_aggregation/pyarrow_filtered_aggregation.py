"""PyArrow implementation for filtered aggregation feature groups."""

from __future__ import annotations

from typing import Any, Optional, Set, Type, Union

import pyarrow as pa
import pyarrow.compute as pc

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.filtered_aggregation.base import (
    FilteredAggregationFeatureGroup,
)

_IDX_COL = "__mloda_fa_idx__"

_PA_AGG_FUNCS: dict[str, str] = {
    "sum": "sum",
    "avg": "mean",
    "mean": "mean",
    "count": "count",
    "min": "min",
    "max": "max",
}


class PyArrowFilteredAggregation(FilteredAggregationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PyArrowTable}

    @classmethod
    def _compute_filtered(
        cls,
        table: pa.Table,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
        filter_column: str,
        filter_value: Any,
    ) -> pa.Table:
        num_rows = table.num_rows
        pa_func = _PA_AGG_FUNCS[agg_type]

        # Mask source column: keep value only where filter matches, null otherwise.
        # pc.equal returns null for null inputs; fill_null(False) treats those as non-matching.
        filter_mask = pc.equal(table.column(filter_column), pa.scalar(filter_value))
        filter_mask = pc.fill_null(filter_mask, False)

        source_type = table.column(source_col).type
        null_scalar = pa.scalar(None, type=source_type)
        masked_col = pc.if_else(filter_mask, table.column(source_col), null_scalar)

        # Build temp table with masked source, partition keys, and index tracker.
        masked_name = "__mloda_fa_src__"
        col_dict: dict[str, Any] = {col: table.column(col) for col in partition_by}
        col_dict[masked_name] = masked_col
        col_dict[_IDX_COL] = pa.array(range(num_rows))
        temp = pa.table(col_dict)

        # Group by partition keys and aggregate on the masked column.
        grouped = temp.group_by(partition_by).aggregate(
            [
                (masked_name, pa_func),
                (_IDX_COL, "list"),
            ]
        )

        agg_col = f"{masked_name}_{pa_func}"
        idx_list_col = f"{_IDX_COL}_list"

        # Broadcast aggregated value back to all original rows.
        result_values: list[Any] = [None] * num_rows
        for g in range(grouped.num_rows):
            agg_val = grouped.column(agg_col)[g].as_py()
            indices = grouped.column(idx_list_col)[g].as_py()
            for idx in indices:
                result_values[idx] = agg_val

        return table.append_column(feature_name, pa.array(result_values))
