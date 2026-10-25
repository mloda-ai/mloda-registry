"""Pure Python reference for aggregation: the shared loop with a kernel-backed compute.

pa.Table is only the I/O container; grouping and aggregation run on ``to_pylist()`` lists.
"""

from __future__ import annotations

import pyarrow as pa
from mloda.provider import FeatureSet

from mloda.community.feature_groups.data_operations.aggregation.base import (
    AggregationFeatureGroup,
    AggregationSpec,
)
from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error
from mloda.community.feature_groups.data_operations.mask_utils import apply_pyarrow_mask
from mloda.testing.feature_groups.data_operations.aggregation_helpers import (
    SUPPORTED_AGG_TYPES,
    aggregate,
    declared_type,
    group_rows,
)


def _compute_group(table: pa.Table, specs: list[AggregationSpec], partition_by: list[str]) -> pa.Table:
    """One row per group (first-occurrence order): the partition columns plus one column per spec."""
    for spec in specs:
        if spec.agg_type not in SUPPORTED_AGG_TYPES:
            raise unsupported_agg_type_error(spec.agg_type, SUPPORTED_AGG_TYPES, framework="Reference")

    keys = [table.column(c).to_pylist() for c in partition_by]
    groups = group_rows(keys, table.num_rows)
    arrays = [
        pa.array([keys[j][rows[0]] for rows in groups.values()], type=table.schema.field(c).type)
        for j, c in enumerate(partition_by)
    ]
    for spec in specs:
        out_type = declared_type(spec.agg_type, table.schema.field(spec.source_col).type)
        source = (
            apply_pyarrow_mask(table, spec.source_col, spec.mask_spec)
            if spec.mask_spec is not None
            else table.column(spec.source_col)
        )
        values = source.to_pylist()
        results = [aggregate([values[i] for i in rows], spec.agg_type) for rows in groups.values()]
        arrays.append(pa.array(results, type=out_type))
    return pa.table(arrays, names=[*partition_by, *(spec.feature_name for spec in specs)])


def reference_aggregation(table: pa.Table, features: FeatureSet) -> pa.Table:
    """Run the shared aggregation loop with the pure Python reference compute."""
    return AggregationFeatureGroup.aggregate(table, features, compute_group=_compute_group)
