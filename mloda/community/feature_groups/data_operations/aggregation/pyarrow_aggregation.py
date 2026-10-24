"""PyArrow implementation for aggregation feature groups.

Uses PyArrow's native ``Table.group_by().aggregate()`` API for vectorized,
C++-backed aggregation.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.aggregation.base import (
    AggregationFeatureGroup,
    AggregationSpec,
)
from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error
from mloda.community.feature_groups.data_operations.helper_columns import unique_helper_name
from mloda.community.feature_groups.data_operations.mask_utils import apply_pyarrow_mask

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

_SUPPORTED_AGG_TYPES = {*_PA_AGG_FUNCS, *_VARIANCE_FUNCS, *_ORDERED_FUNCS}


class PyArrowAggregation(Implementation[AggregationFeatureGroup]):
    feature_group = AggregationFeatureGroup
    compute_framework = PyArrowTable
    supported = frozenset(_SUPPORTED_AGG_TYPES)

    @classmethod
    def calculate(
        cls, group: type[AggregationFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.aggregate(data, features, compute_group=cls._compute_group)

    @classmethod
    def _compute_group(cls, table: pa.Table, specs: list[AggregationSpec], partition_by: list[str]) -> pa.Table:
        for spec in specs:
            if spec.agg_type not in _SUPPORTED_AGG_TYPES:
                raise unsupported_agg_type_error(spec.agg_type, _SUPPORTED_AGG_TYPES, framework="PyArrow")

        # Each masked source is a copy under a temp name, its mask built from the original table.
        taken = set(table.column_names) | {spec.feature_name for spec in specs}
        entries: list[tuple[Any, ...]] = []
        for spec in specs:
            source = spec.source_col
            if spec.mask_spec is not None:
                masked = apply_pyarrow_mask(table, spec.source_col, spec.mask_spec).column(spec.source_col)
                source = unique_helper_name("__mloda_masked_src__", taken)
                taken.add(source)
                table = table.append_column(source, masked)

            if spec.agg_type in _PA_AGG_FUNCS:
                entries.append((source, _PA_AGG_FUNCS[spec.agg_type]))
            elif spec.agg_type in _VARIANCE_FUNCS:
                pa_func, ddof = _VARIANCE_FUNCS[spec.agg_type]
                entries.append((source, pa_func, pc.VarianceOptions(ddof=ddof)))
            else:
                entries.append((source, _ORDERED_FUNCS[spec.agg_type]))

        ordered = any(spec.agg_type in _ORDERED_FUNCS for spec in specs)
        grouped = table.group_by(partition_by, use_threads=not ordered).aggregate(entries)

        # Rename aggregate columns by position (auto names collide for a shared source + func).
        # Newer PyArrow puts the keys first, older puts them last.
        keys_first = grouped.column_names[: len(partition_by)] == partition_by
        offset = len(partition_by) if keys_first else 0
        names = list(grouped.column_names)
        for i, spec in enumerate(specs):
            names[offset + i] = spec.feature_name
        return grouped.rename_columns(names)
