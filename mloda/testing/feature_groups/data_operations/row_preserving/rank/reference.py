"""Test reference implementation for rank feature groups.

Accepts PyArrow tables but computes in Python. Used as the cross-framework
comparison baseline in test suites.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mloda.provider import FeatureSet

from mloda.testing.feature_groups.data_operations.aggregation_helpers import order_key, ordered_groups

if TYPE_CHECKING:
    import pyarrow as pa


def _compute_rank(
    table: pa.Table,
    feature_name: str,
    partition_by: list[str],
    order_by: str,
    rank_type: str,
) -> pa.Table:
    import pyarrow as pa

    num_rows = table.num_rows
    keys = [table.column(col).to_pylist() for col in partition_by]
    order_vals = table.column(order_by).to_pylist()

    def run_starts(rows: list[int]) -> list[int]:
        """Position of the first row of each tie run, repeated for every row in the run."""
        starts: list[int] = []
        for pos, idx in enumerate(rows):
            if pos > 0 and order_key(order_vals[idx]) == order_key(order_vals[rows[pos - 1]]):
                starts.append(starts[-1])
            else:
                starts.append(pos)
        return starts

    result_values: list[Any] = [None] * num_rows
    declared = pa.int64()
    if rank_type == "percent_rank":
        declared = pa.float64()
    elif rank_type.startswith(("top_", "bottom_")):
        declared = pa.bool_()

    for rows in ordered_groups(keys, order_vals, num_rows):
        n = len(rows)

        if rank_type == "row_number":
            for pos, idx in enumerate(rows):
                result_values[idx] = pos + 1

        elif rank_type == "rank":
            for start, idx in zip(run_starts(rows), rows):
                result_values[idx] = start + 1

        elif rank_type == "dense_rank":
            dense = 0
            prev = -1
            for start, idx in zip(run_starts(rows), rows):
                if start != prev:
                    dense += 1
                    prev = start
                result_values[idx] = dense

        elif rank_type == "percent_rank":
            for start, idx in zip(run_starts(rows), rows):
                result_values[idx] = 0.0 if n == 1 else start / (n - 1)

        elif rank_type.startswith("ntile_"):
            ntile_n = int(rank_type[len("ntile_") :])
            for pos, idx in enumerate(rows):
                result_values[idx] = (pos * ntile_n) // n + 1

        elif rank_type.startswith("top_"):
            top_n = int(rank_type[len("top_") :])
            real = [i for i in rows if order_key(order_vals[i])[0] == 0]
            others = [i for i in rows if order_key(order_vals[i])[0] != 0]
            for pos, idx in enumerate(sorted(real, key=lambda i: order_vals[i], reverse=True) + others):
                result_values[idx] = pos + 1 <= top_n

        elif rank_type.startswith("bottom_"):
            bottom_n = int(rank_type[len("bottom_") :])
            for pos, idx in enumerate(rows):
                result_values[idx] = pos + 1 <= bottom_n

        else:
            raise ValueError(f"Unsupported rank type: {rank_type}")

    return table.append_column(feature_name, pa.array(result_values, type=declared))


def reference_rank(group: Any, table: pa.Table, features: FeatureSet) -> pa.Table:
    """Run the group's rank loop with the PyArrow-backed reference compute."""
    return group.rank(table, features, compute_rank=_compute_rank)
