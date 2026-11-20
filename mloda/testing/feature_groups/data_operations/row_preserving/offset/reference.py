"""Test reference implementation for offset feature groups.

Accepts PyArrow tables but computes in Python. Used as the cross-framework
comparison baseline in test suites.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mloda.provider import FeatureSet

from mloda.testing.feature_groups.data_operations.aggregation_helpers import ordered_groups

if TYPE_CHECKING:
    import pyarrow as pa


def _compute_offset(
    table: pa.Table,
    feature_name: str,
    source_col: str,
    partition_by: list[str],
    order_by: str,
    offset_type: str,
) -> pa.Table:
    import pyarrow as pa

    num_rows = table.num_rows

    keys = [table.column(col).to_pylist() for col in partition_by]
    order_vals = table.column(order_by).to_pylist()
    source_vals = table.column(source_col).to_pylist()

    result_values: list[Any] = [None] * num_rows

    for rows in ordered_groups(keys, order_vals, num_rows):
        n = len(rows)
        vals = [source_vals[i] for i in rows]

        if offset_type.startswith("lag_"):
            offset_n = int(offset_type[len("lag_") :])
            for pos in range(n):
                idx = rows[pos]
                if pos >= offset_n:
                    result_values[idx] = vals[pos - offset_n]

        elif offset_type.startswith("lead_"):
            offset_n = int(offset_type[len("lead_") :])
            for pos in range(n):
                idx = rows[pos]
                if pos + offset_n < n:
                    result_values[idx] = vals[pos + offset_n]

        elif offset_type.startswith("diff_"):
            offset_n = int(offset_type[len("diff_") :])
            for pos in range(n):
                idx = rows[pos]
                curr = vals[pos]
                if pos >= offset_n and curr is not None and vals[pos - offset_n] is not None:
                    result_values[idx] = curr - vals[pos - offset_n]

        elif offset_type.startswith("pct_change_"):
            offset_n = int(offset_type[len("pct_change_") :])
            for pos in range(n):
                idx = rows[pos]
                curr = vals[pos]
                prev = vals[pos - offset_n] if pos >= offset_n else None
                if curr is not None and prev is not None and prev != 0:
                    result_values[idx] = (curr - prev) / prev

        elif offset_type == "first_value":
            first = next((v for v in vals if v is not None), None)
            for pos in range(n):
                result_values[rows[pos]] = first

        elif offset_type == "last_value":
            last = next((v for v in reversed(vals) if v is not None), None)
            for pos in range(n):
                result_values[rows[pos]] = last

        else:
            raise ValueError(f"Unsupported offset type: {offset_type}")

    declared = pa.float64() if offset_type.startswith("pct_change_") else table.schema.field(source_col).type
    return table.append_column(feature_name, pa.array(result_values, type=declared))


def reference_offset(group: Any, table: pa.Table, features: FeatureSet) -> pa.Table:
    """Run the group's offset loop with the PyArrow-backed reference compute."""
    return group.offset(table, features, compute_offset=_compute_offset)
