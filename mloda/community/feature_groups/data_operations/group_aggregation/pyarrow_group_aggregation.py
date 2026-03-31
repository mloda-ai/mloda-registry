"""PyArrow implementation for group aggregation feature groups.

Uses PyArrow's native ``Table.group_by().aggregate()`` API (available since
PyArrow 12) for vectorized, C++-backed group aggregation. Median and mode
fall back to a list-collect-then-compute path because PyArrow has no exact
grouped median or grouped mode function.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Set, Type, Union

import pyarrow as pa
import pyarrow.compute as pc

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.group_aggregation.base import (
    GroupAggregationFeatureGroup,
)

# Aggregation types with direct PyArrow group_by support.
_PA_AGG_FUNCS: dict[str, str] = {
    "sum": "sum",
    "avg": "mean",
    "count": "count",
    "min": "min",
    "max": "max",
    "nunique": "count_distinct",
}

# Sample statistics need VarianceOptions(ddof=1).
_SAMPLE_STAT_FUNCS: dict[str, str] = {
    "std": "stddev",
    "var": "variance",
}

# Ordered aggregates need use_threads=False in PyArrow.
_ORDERED_FUNCS: dict[str, str] = {
    "first": "first",
    "last": "last",
}


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
        if agg_type in _PA_AGG_FUNCS:
            pa_func = _PA_AGG_FUNCS[agg_type]
            grouped = table.group_by(partition_by).aggregate([(source_col, pa_func)])
        elif agg_type in _SAMPLE_STAT_FUNCS:
            pa_func = _SAMPLE_STAT_FUNCS[agg_type]
            grouped = table.group_by(partition_by).aggregate([(source_col, pa_func, pc.VarianceOptions(ddof=1))])
        elif agg_type in _ORDERED_FUNCS:
            pa_func = _ORDERED_FUNCS[agg_type]
            grouped = table.group_by(partition_by, use_threads=False).aggregate([(source_col, pa_func)])
        elif agg_type in ("median", "mode"):
            return cls._compute_via_list(table, feature_name, source_col, partition_by, agg_type)
        else:
            raise ValueError(f"Unsupported aggregation type: {agg_type}")

        # Rename auto-generated column (e.g. "val_sum") to feature_name.
        auto_col = f"{source_col}_{pa_func}"
        names = [feature_name if c == auto_col else c for c in grouped.column_names]
        return grouped.rename_columns(names)

    @classmethod
    def _compute_via_list(
        cls,
        table: pa.Table,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
    ) -> pa.Table:
        """Collect values per group via native list aggregation, then reduce in Python.

        This avoids row-by-row .as_py() calls: PyArrow builds the per-group
        lists in C++, and we only cross into Python once per group.
        """
        grouped = table.group_by(partition_by).aggregate([(source_col, "list")])
        list_col = f"{source_col}_list"
        agg_values: list[Any] = []

        for vals in grouped.column(list_col).to_pylist():
            non_null = [v for v in vals if v is not None]
            if not non_null:
                agg_values.append(None)
            elif agg_type == "median":
                agg_values.append(cls._median(non_null))
            else:
                agg_values.append(cls._mode(non_null))

        arrays = [grouped.column(col) for col in partition_by]
        arrays.append(pa.array(agg_values))
        return pa.table(arrays, names=list(partition_by) + [feature_name])

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
