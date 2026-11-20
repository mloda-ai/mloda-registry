"""Test reference implementation for sessionization feature groups, pure Python over PyArrow tables."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from mloda.provider import FeatureSet

from mloda.testing.feature_groups.data_operations.aggregation_helpers import group_rows, order_key

if TYPE_CHECKING:
    import pyarrow as pa


def _compute_session(
    table: pa.Table,
    feature_name: str,
    order_col: str,
    threshold_seconds: int,
    partition_by: list[str],
) -> pa.Table:
    import pyarrow as pa

    order_type = table.column(order_col).type
    if not pa.types.is_timestamp(order_type):
        raise ValueError(f"Order column {order_col!r} must be a timestamp for sessionization; got {order_type}.")

    keys = [table.column(col).to_pylist() for col in partition_by]
    order_vals = table.column(order_col).to_pylist()
    threshold = timedelta(seconds=threshold_seconds)

    groups = list(group_rows(keys, table.num_rows).values())
    groups.sort(key=lambda rows: tuple(order_key(col[rows[0]]) for col in keys) if rows else ())

    result: list[Any] = [None] * table.num_rows
    counter = -1
    for rows in groups:
        prev: Any = None
        for pos, i in enumerate(sorted(rows, key=lambda r: order_key(order_vals[r]))):
            cur = order_vals[i]
            if pos == 0 or cur is None or prev is None or cur - prev > threshold:
                counter += 1
            result[i] = counter
            prev = cur

    return table.append_column(feature_name, pa.array(result, type=pa.int64()))


def reference_sessionization(group: Any, table: pa.Table, features: FeatureSet) -> pa.Table:
    """Run the group's sessionize loop with the pure Python reference compute."""
    return group.sessionize(table, features, compute_session=_compute_session)
