"""PyArrow implementation for window aggregation feature groups."""

from __future__ import annotations

from typing import Any, Optional, Set, Type, Union

import pyarrow as pa

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.pyarrow_aggregation_helpers import aggregate
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
    WindowAggregationFeatureGroup,
)


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

        # Build group keys per row (using Python objects so None is a valid key)
        keys: list[tuple[Any, ...]] = []
        for i in range(num_rows):
            key = tuple(table.column(col)[i].as_py() for col in partition_by)
            keys.append(key)

        # Collect source values per group, preserving row indices
        groups: dict[tuple[Any, ...], list[tuple[Any, Any]]] = {}
        for i in range(num_rows):
            key = keys[i]
            val = table.column(source_col)[i].as_py()
            order_val = table.column(order_by)[i].as_py() if order_by else i
            if key not in groups:
                groups[key] = []
            groups[key].append((order_val, val))

        # Compute aggregate per group
        agg_results: dict[tuple[Any, ...], Any] = {}
        for key, pairs in groups.items():
            values = [v for _, v in pairs]
            if agg_type in ("first", "last") and order_by:
                sorted_vals = [
                    v for _, v in sorted(pairs, key=lambda p: (p[0] is None, p[0] if p[0] is not None else 0))
                ]
                agg_results[key] = aggregate(sorted_vals, agg_type)
            else:
                agg_results[key] = aggregate(values, agg_type)

        # Broadcast back to every row
        result_values = [agg_results[keys[i]] for i in range(num_rows)]

        new_col = pa.array(result_values)
        return table.append_column(feature_name, new_col)
