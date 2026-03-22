"""PyArrow implementation for group aggregation feature groups."""

from __future__ import annotations

from collections import Counter
from typing import Any, Set, Type, Union

import pyarrow as pa

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.group_aggregation.base import (
    GroupAggregationFeatureGroup,
)


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
            result_values.append(cls._aggregate(values, agg_type))

        # Build output table: partition columns + aggregated column
        arrays = [pa.array(partition_columns[col]) for col in partition_by]
        arrays.append(pa.array(result_values))
        names = list(partition_by) + [feature_name]
        return pa.table(arrays, names=names)

    @classmethod
    def _aggregate(cls, values: list[Any], agg_type: str) -> Any:
        """Compute a single aggregate over a list of values (may contain None)."""
        non_null = [v for v in values if v is not None]

        if not non_null:
            return None

        if agg_type == "sum":
            return sum(non_null)
        if agg_type == "avg":
            return sum(non_null) / len(non_null)
        if agg_type == "count":
            return len(non_null)
        if agg_type == "min":
            return min(non_null)
        if agg_type == "max":
            return max(non_null)
        if agg_type == "std":
            return cls._std(non_null)
        if agg_type == "var":
            return cls._var(non_null)
        if agg_type == "median":
            return cls._median(non_null)
        if agg_type == "mode":
            return cls._mode(non_null)
        if agg_type == "nunique":
            return len(set(non_null))
        if agg_type == "first":
            return non_null[0]
        if agg_type == "last":
            return non_null[-1]

        raise ValueError(f"Unsupported aggregation type: {agg_type}")

    @classmethod
    def _std(cls, values: list[Any]) -> Any:
        if len(values) < 2:
            return None
        return cls._var(values) ** 0.5

    @classmethod
    def _var(cls, values: list[Any]) -> Any:
        if len(values) < 2:
            return None
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / (len(values) - 1)

    @classmethod
    def _median(cls, values: list[Any]) -> Any:
        s = sorted(values)
        n = len(s)
        mid = n // 2
        if n % 2 == 0:
            return (s[mid - 1] + s[mid]) / 2.0
        return float(s[mid])

    @classmethod
    def _mode(cls, values: list[Any]) -> Any:
        if not values:
            return None
        counts = Counter(values)
        return counts.most_common(1)[0][0]
