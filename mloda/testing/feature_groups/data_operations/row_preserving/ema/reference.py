"""Test reference implementation for EMA feature groups, pure Python over PyArrow tables."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mloda.provider import FeatureSet

from mloda.testing.feature_groups.data_operations.aggregation_helpers import ordered_groups

if TYPE_CHECKING:
    import pyarrow as pa


def _compute_ema(
    table: pa.Table,
    feature_name: str,
    source_col: str,
    span: int,
    partition_by: list[str],
    order_by: str,
) -> pa.Table:
    import pyarrow as pa

    keys = [table.column(col).to_pylist() for col in partition_by]
    order_vals = table.column(order_by).to_pylist()
    source_vals = table.column(source_col).to_pylist()
    alpha = 2.0 / (span + 1)

    result: list[Any] = [None] * table.num_rows
    for rows in ordered_groups(keys, order_vals, table.num_rows):
        state: Any = None
        for i in rows:
            x = source_vals[i]
            if x is None:
                continue
            state = float(x) if state is None else alpha * x + (1 - alpha) * state
            result[i] = state

    return table.append_column(feature_name, pa.array(result, type=pa.float64()))


def reference_ema(group: Any, table: pa.Table, features: FeatureSet) -> pa.Table:
    """Run the group's EMA loop with the pure Python reference compute."""
    return group.ema(table, features, compute_ema=_compute_ema)
