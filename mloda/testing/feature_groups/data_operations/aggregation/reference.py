"""Pure Python reference for aggregation: runs the loop of the group passed in with a kernel-backed compute.

pa.Table is only the I/O container; grouping and aggregation run on ``to_pylist()`` lists.
"""

from __future__ import annotations

from typing import Any, Protocol

import pyarrow as pa
from mloda.provider import FeatureSet

from mloda.testing.feature_groups.data_operations.aggregation_helpers import (
    SUPPORTED_AGG_TYPES,
    aggregate,
    declared_type,
    group_rows,
    unsupported_agg_type_error,
)
from mloda.testing.feature_groups.data_operations.helpers import masked_column


class AggregationSpec(Protocol):
    """One aggregation as the group's loop hands it to the compute."""

    @property
    def feature_name(self) -> str: ...

    @property
    def source_col(self) -> str: ...

    @property
    def agg_type(self) -> str: ...

    @property
    def mask_spec(self) -> list[tuple[str, str, Any]] | None: ...


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
            masked_column(table, spec.source_col, spec.mask_spec)
            if spec.mask_spec is not None
            else table.column(spec.source_col)
        )
        values = source.to_pylist()
        results = [aggregate([values[i] for i in rows], spec.agg_type) for rows in groups.values()]
        arrays.append(pa.array(results, type=out_type))
    return pa.table(arrays, names=[*partition_by, *(spec.feature_name for spec in specs)])


def reference_aggregation(group: Any, table: pa.Table, features: FeatureSet) -> pa.Table:
    """Run the group's aggregation loop with the pure Python reference compute."""
    return group.aggregate(table, features, compute_group=_compute_group)
