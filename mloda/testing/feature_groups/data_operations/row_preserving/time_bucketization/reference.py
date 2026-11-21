"""Pure Python reference for time bucketization: runs the group's loop with a Python compute.

Math runs on the wall-clock value of the column's own zone; tzinfo is re-attached at the end.
Only ms and us timestamps with no zone, UTC or a fixed offset are in scope.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mloda.provider import FeatureSet

from mloda.testing.feature_groups.data_operations.row_preserving.bucket_helpers import (
    assert_timestamp,
    bucket_timestamp,
)

if TYPE_CHECKING:
    import pyarrow as pa

TIME_BUCKET_OP_NAMES = frozenset({"floor", "ceil", "round"})


def _compute(table: pa.Table, feature_name: str, source_col: str, op: str, n: int, unit: str) -> pa.Table:
    import pyarrow as pa

    if op not in TIME_BUCKET_OP_NAMES:
        raise ValueError(f"Unsupported bucket op {op!r}. Supported: {sorted(TIME_BUCKET_OP_NAMES)}")
    column = table.column(source_col)
    values = [None if v is None else bucket_timestamp(v, op, n, unit) for v in column.to_pylist()]
    return table.append_column(feature_name, pa.array(values, type=column.type))


def reference_time_bucketization(group: Any, table: pa.Table, features: FeatureSet) -> pa.Table:
    """The table plus one pure Python column per bucketization feature."""
    return group.bucketize(table, features, assert_timestamp=assert_timestamp, compute_bucket=_compute)
