"""Test reference implementation for ffill feature groups, pure Python over PyArrow tables."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mloda.provider import FeatureSet

from mloda.testing.feature_groups.data_operations.aggregation_helpers import ordered_groups

if TYPE_CHECKING:
    import pyarrow as pa


def _compute_ffill(
    table: pa.Table,
    feature_name: str,
    source_col: str,
    partition_by: list[str],
    order_by: str,
) -> pa.Table:
    import pyarrow as pa

    keys = [table.column(col).to_pylist() for col in partition_by]
    order_vals = table.column(order_by).to_pylist()
    source_vals = table.column(source_col).to_pylist()

    result: list[Any] = [None] * table.num_rows
    for rows in ordered_groups(keys, order_vals, table.num_rows):
        last: Any = None
        for i in rows:
            if source_vals[i] is not None:
                last = source_vals[i]
            result[i] = last

    declared = table.schema.field(source_col).type
    return table.append_column(feature_name, pa.array(result, type=declared))


def reference_ffill(group: Any, table: pa.Table, features: FeatureSet) -> pa.Table:
    """Run the group's ffill loop with the pure Python reference compute."""
    return group.ffill(table, features, compute_ffill=_compute_ffill)
