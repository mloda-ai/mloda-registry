"""PyArrow implementation for aggregation feature groups.

Uses PyArrow's native ``Table.group_by().aggregate()`` API for vectorized,
C++-backed aggregation.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.aggregation.base import (
    AggregationFeatureGroup,
)

# Aggregation types with direct PyArrow group_by support.
_PA_AGG_FUNCS: dict[str, str] = {
    "sum": "sum",
    "avg": "mean",
    "mean": "mean",
    "count": "count",
    "min": "min",
    "max": "max",
    "nunique": "count_distinct",
}

# Variance/stddev operations mapped to (PyArrow func name, ddof).
_VARIANCE_FUNCS: dict[str, tuple[str, int]] = {
    "std": ("stddev", 0),
    "var": ("variance", 0),
    "std_pop": ("stddev", 0),
    "std_samp": ("stddev", 1),
    "var_pop": ("variance", 0),
    "var_samp": ("variance", 1),
}

# Ordered aggregates need use_threads=False in PyArrow.
_ORDERED_FUNCS: dict[str, str] = {
    "first": "first",
    "last": "last",
}


class PyArrowAggregation(AggregationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
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
        elif agg_type in _VARIANCE_FUNCS:
            pa_func, ddof = _VARIANCE_FUNCS[agg_type]
            grouped = table.group_by(partition_by).aggregate([(source_col, pa_func, pc.VarianceOptions(ddof=ddof))])
        elif agg_type in _ORDERED_FUNCS:
            pa_func = _ORDERED_FUNCS[agg_type]
            grouped = table.group_by(partition_by, use_threads=False).aggregate([(source_col, pa_func)])
        else:
            raise ValueError(f"Unsupported aggregation type: {agg_type}")

        # Rename auto-generated column (e.g. "val_sum") to feature_name.
        auto_col = f"{source_col}_{pa_func}"
        names = [feature_name if c == auto_col else c for c in grouped.column_names]
        return grouped.rename_columns(names)
