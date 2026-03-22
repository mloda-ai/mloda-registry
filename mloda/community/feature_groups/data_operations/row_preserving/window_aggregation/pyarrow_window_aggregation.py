"""PyArrow implementation for window aggregation feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

import pyarrow as pa

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
    WindowAggregationFeatureGroup,
)


class PyArrowWindowAggregation(WindowAggregationFeatureGroup):
    """Reference implementation using dict-based grouping on PyArrow tables."""

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
    ) -> pa.Table:
        num_rows = table.num_rows

        # Build group keys per row (using Python objects so None is a valid key)
        keys: list[tuple[Any, ...]] = []
        for i in range(num_rows):
            key = tuple(table.column(col)[i].as_py() for col in partition_by)
            keys.append(key)

        # Collect source values per group, preserving order
        groups: dict[tuple[Any, ...], list[Any]] = {}
        for i in range(num_rows):
            key = keys[i]
            val = table.column(source_col)[i].as_py()
            if key not in groups:
                groups[key] = []
            groups[key].append(val)

        # Compute aggregate per group
        agg_results: dict[tuple[Any, ...], Any] = {}
        for key, values in groups.items():
            agg_results[key] = cls._aggregate(values, agg_type)

        # Broadcast back to every row
        result_values = [agg_results[keys[i]] for i in range(num_rows)]

        new_col = pa.array(result_values)
        return table.append_column(feature_name, new_col)
