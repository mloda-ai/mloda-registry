"""Pure Python reference for scalar aggregate: the shared loop with a kernel-backed compute.

pa.Table is only the I/O container; the aggregate runs on a ``to_pylist()`` list.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
from mloda.provider import FeatureSet

from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error
from mloda.community.feature_groups.data_operations.mask_utils import apply_pyarrow_mask
from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.base import (
    ScalarAggregateFeatureGroup,
)
from mloda.testing.feature_groups.data_operations.aggregation_helpers import aggregate, declared_type


def _compute_aggregation(
    table: pa.Table,
    feature_name: str,
    source_col: str,
    agg_type: str,
    mask_spec: list[tuple[str, str, Any]] | None = None,
) -> pa.Table:
    supported = ScalarAggregateFeatureGroup._SUPPORTED_AGG_TYPES
    if agg_type not in supported:
        raise unsupported_agg_type_error(agg_type, supported, framework="Reference")

    source = apply_pyarrow_mask(table, source_col, mask_spec) if mask_spec is not None else table.column(source_col)
    value = aggregate(source.to_pylist(), agg_type)
    out_type = declared_type(agg_type, table.schema.field(source_col).type)
    return table.append_column(feature_name, pa.array([value] * table.num_rows, type=out_type))


def reference_scalar_aggregate(table: pa.Table, features: FeatureSet) -> pa.Table:
    """Run the shared scalar aggregate loop with the pure Python reference compute."""
    return ScalarAggregateFeatureGroup.aggregate(table, features, compute_aggregation=_compute_aggregation)
