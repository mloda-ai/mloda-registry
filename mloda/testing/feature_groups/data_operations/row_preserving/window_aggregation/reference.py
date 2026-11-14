"""Pure Python reference for window aggregation: runs the loop of the group passed in with a kernel-backed compute.

pa.Table is only the I/O container; grouping and aggregation run on ``to_pylist()`` lists.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
from mloda.provider import FeatureSet

from mloda.testing.feature_groups.data_operations.aggregation_helpers import (
    SUPPORTED_AGG_TYPES,
    aggregate,
    declared_type,
    group_rows,
    order_key,
    unsupported_agg_type_error,
)
from mloda.testing.feature_groups.data_operations.helpers import masked_column


def _compute_window(
    table: pa.Table,
    feature_name: str,
    source_col: str,
    partition_by: list[str],
    agg_type: str,
    order_by: str | None = None,
    mask_spec: list[tuple[str, str, Any]] | None = None,
) -> pa.Table:
    if agg_type not in SUPPORTED_AGG_TYPES:
        raise unsupported_agg_type_error(agg_type, SUPPORTED_AGG_TYPES, framework="Reference")

    keys = [table.column(c).to_pylist() for c in partition_by]
    # Keys and order stay unmasked; only the aggregated values are masked.
    source = masked_column(table, source_col, mask_spec) if mask_spec is not None else table.column(source_col)
    values = source.to_pylist()
    order_vals = table.column(order_by).to_pylist() if order_by else None

    out_type = declared_type(agg_type, table.schema.field(source_col).type)
    result: list[Any] = [None] * table.num_rows
    for rows in group_rows(keys, table.num_rows).values():
        if order_vals is not None and agg_type in ("first", "last"):
            rows = sorted(rows, key=lambda i: order_key(order_vals[i]))
        value = aggregate([values[i] for i in rows], agg_type)
        for i in rows:
            result[i] = value
    return table.append_column(feature_name, pa.array(result, type=out_type))


def reference_window_aggregation(group: Any, table: pa.Table, features: FeatureSet) -> pa.Table:
    """Run the group's window aggregation loop with the pure Python reference compute."""
    return group.aggregate(table, features, compute_window=_compute_window)
