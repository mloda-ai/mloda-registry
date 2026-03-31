"""PyArrow implementation for group aggregation feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

import pyarrow as pa

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.group_aggregation.base import (
    GroupAggregationFeatureGroup,
)
from mloda.community.feature_groups.data_operations.pyarrow_aggregation_helpers import aggregate


class PyArrowGroupAggregation(GroupAggregationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PyArrowTable}

    @classmethod
    def _compute_group(
        cls,
        table: pa.Table,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
    ) -> pa.Table:
        # PyArrow has no native group-by-and-reduce API that returns a reduced
        # table, so this implementation uses Python dict-based grouping with
        # row-by-row .as_py() calls. Other frameworks (DuckDB, Polars, Pandas,
        # SQLite) use their native aggregation APIs.
        num_rows = table.num_rows

        # Build group keys per row (using Python objects so None is a valid key)
        keys: list[tuple[Any, ...]] = []
        for i in range(num_rows):
            key = tuple(table.column(col)[i].as_py() for col in partition_by)
            keys.append(key)

        # Collect source values per group, preserving insertion order
        groups: dict[tuple[Any, ...], list[Any]] = {}
        for i in range(num_rows):
            key = keys[i]
            val = table.column(source_col)[i].as_py()
            if key not in groups:
                groups[key] = []
            groups[key].append(val)

        # Compute aggregate per group and build output columns
        partition_columns: dict[str, list[Any]] = {col: [] for col in partition_by}
        result_values: list[Any] = []

        for key, values in groups.items():
            for j, col in enumerate(partition_by):
                partition_columns[col].append(key[j])
            result_values.append(aggregate(values, agg_type))

        # Build output table: partition columns + aggregated column
        arrays = [pa.array(partition_columns[col]) for col in partition_by]
        arrays.append(pa.array(result_values))
        names = list(partition_by) + [feature_name]
        return pa.table(arrays, names=names)
