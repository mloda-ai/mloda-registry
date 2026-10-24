"""Test reference implementation for aggregation feature groups.

Accepts PyArrow tables but computes in Python. Used as the cross-framework
comparison baseline in test suites. Uses PyArrow's native
``Table.group_by().aggregate()`` API for vectorized aggregation where
possible. Median and mode fall back to a list-collect-then-compute path
because PyArrow has no exact grouped median or grouped mode function.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from mloda.provider import FeatureSet

from mloda.community.feature_groups.data_operations.aggregation.base import (
    AggregationFeatureGroup,
    AggregationSpec,
)
from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error
from mloda.community.feature_groups.data_operations.mask_utils import apply_pyarrow_mask
from mloda.community.feature_groups.data_operations.pyarrow_helpers import rename_aggregates_by_position
from mloda.testing.feature_groups.data_operations import aggregation_helpers

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

_SUPPORTED_AGG_TYPES = {*_PA_AGG_FUNCS, *_VARIANCE_FUNCS, *_ORDERED_FUNCS, "median", "mode"}


def _compute_single(table: pa.Table, spec: AggregationSpec, partition_by: list[str]) -> pa.Table:
    feature_name, source_col, agg_type, mask_spec = spec
    if mask_spec is not None:
        table = apply_pyarrow_mask(table, source_col, mask_spec)

    if agg_type in _PA_AGG_FUNCS:
        pa_func = _PA_AGG_FUNCS[agg_type]
        grouped = table.group_by(partition_by).aggregate([(source_col, pa_func)])
    elif agg_type in _VARIANCE_FUNCS:
        pa_func, ddof = _VARIANCE_FUNCS[agg_type]
        grouped = table.group_by(partition_by).aggregate([(source_col, pa_func, pc.VarianceOptions(ddof=ddof))])
    elif agg_type in _ORDERED_FUNCS:
        pa_func = _ORDERED_FUNCS[agg_type]
        grouped = table.group_by(partition_by, use_threads=False).aggregate([(source_col, pa_func)])
    elif agg_type in ("median", "mode"):
        return _compute_via_list(table, feature_name, source_col, partition_by, agg_type)
    else:
        raise unsupported_agg_type_error(agg_type, _SUPPORTED_AGG_TYPES, framework="Reference")

    return rename_aggregates_by_position(grouped, partition_by, [feature_name])


def _compute_via_list(
    table: pa.Table,
    feature_name: str,
    source_col: str,
    partition_by: list[str],
    agg_type: str,
) -> pa.Table:
    """Collect values per group via native list aggregation, then reduce in Python.

    This avoids row-by-row .as_py() calls: PyArrow builds the per-group
    lists in C++, and we only cross into Python once per group.

    ``partition_by=[]`` bypasses ``group_by([]).aggregate(..., "list")``, which PyArrow
    rejects, and collects the whole column directly instead.
    """
    if not partition_by:
        non_null = [v for v in table.column(source_col).to_pylist() if v is not None]
        if not non_null:
            agg_value = None
        elif agg_type == "median":
            agg_value = aggregation_helpers.median(non_null)
        else:
            agg_value = aggregation_helpers.mode(non_null)
        return pa.table([pa.array([agg_value])], names=[feature_name])

    grouped = table.group_by(partition_by).aggregate([(source_col, "list")])
    list_col = f"{source_col}_list"
    agg_values: list[Any] = []

    for vals in grouped.column(list_col).to_pylist():
        non_null = [v for v in vals if v is not None]
        if not non_null:
            agg_values.append(None)
        elif agg_type == "median":
            agg_values.append(aggregation_helpers.median(non_null))
        else:
            agg_values.append(aggregation_helpers.mode(non_null))

    arrays = [grouped.column(col) for col in partition_by]
    arrays.append(pa.array(agg_values))
    return pa.table(arrays, names=list(partition_by) + [feature_name])


def _group_key(table: pa.Table, partition_by: list[str]) -> list[tuple[Any, ...]]:
    """Per-row partition key tuples; NaN is normalized so equal NaN keys match."""
    if not partition_by:
        return [()] * table.num_rows
    columns = [table.column(c).to_pylist() for c in partition_by]
    return [tuple("nan" if isinstance(v, float) and v != v else v for v in row) for row in zip(*columns)]


def _compute_group(table: pa.Table, specs: list[AggregationSpec], partition_by: list[str]) -> pa.Table:
    """Compute each spec on the original table, then join the results by partition key.

    Groups come in the first spec's order; the output is the partition columns plus one column per spec.
    """
    for spec in specs:
        if spec.agg_type not in _SUPPORTED_AGG_TYPES:
            raise unsupported_agg_type_error(spec.agg_type, _SUPPORTED_AGG_TYPES, framework="Reference")

    results = [_compute_single(table, spec, partition_by) for spec in specs]
    first_keys = _group_key(results[0], partition_by)

    arrays = [results[0].column(c) for c in partition_by]
    for i, (spec, result) in enumerate(zip(specs, results)):
        column = result.column(spec.feature_name)
        if i > 0:
            by_key = dict(zip(_group_key(result, partition_by), column.to_pylist()))
            column = pa.array([by_key[key] for key in first_keys], type=column.type)
        arrays.append(column)
    return pa.table(arrays, names=[*partition_by, *(spec.feature_name for spec in specs)])


def reference_aggregation(table: pa.Table, features: FeatureSet) -> pa.Table:
    """Run the shared aggregation loop with the PyArrow-backed reference compute."""
    return AggregationFeatureGroup.aggregate(table, features, compute_group=_compute_group)
