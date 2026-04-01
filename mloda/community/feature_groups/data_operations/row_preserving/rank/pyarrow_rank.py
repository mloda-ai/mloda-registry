"""PyArrow implementation for rank feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

import pyarrow as pa

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.rank.base import (
    RankFeatureGroup,
)


class PyArrowRank(RankFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PyArrowTable}

    @classmethod
    def _compute_rank(
        cls,
        table: pa.Table,
        feature_name: str,
        partition_by: list[str],
        order_by: str,
        rank_type: str,
    ) -> pa.Table:
        num_rows = table.num_rows

        # Convert columns to Python lists once (O(n) per column) instead of
        # extracting individual elements via .as_py() which is O(n) calls
        # crossing the Python/C++ boundary.
        partition_cols = {col: table.column(col).to_pylist() for col in partition_by}
        order_vals = table.column(order_by).to_pylist()

        # Build group keys per row
        keys: list[tuple[Any, ...]] = []
        for i in range(num_rows):
            key = tuple(partition_cols[col][i] for col in partition_by)
            keys.append(key)

        # Collect (index, order_value) per group
        groups: dict[tuple[Any, ...], list[tuple[int, Any]]] = {}
        for i in range(num_rows):
            key = keys[i]
            val = order_vals[i]
            if key not in groups:
                groups[key] = []
            groups[key].append((i, val))

        # Sort each group by order_by value (nulls last)
        for key in groups:
            groups[key].sort(key=lambda x: (x[1] is None, x[1] if x[1] is not None else 0))

        # Compute rank values
        result_values: list[Any] = [0] * num_rows

        for key, sorted_rows in groups.items():
            n = len(sorted_rows)

            if rank_type == "row_number":
                for pos, (idx, _) in enumerate(sorted_rows):
                    result_values[idx] = pos + 1

            elif rank_type == "rank":
                pos = 0
                while pos < n:
                    # Find run of equal values
                    run_start = pos
                    while pos < n and sorted_rows[pos][1] == sorted_rows[run_start][1]:
                        pos += 1
                    rank_val = run_start + 1
                    for j in range(run_start, pos):
                        result_values[sorted_rows[j][0]] = rank_val

            elif rank_type == "dense_rank":
                dense = 1
                pos = 0
                while pos < n:
                    run_start = pos
                    while pos < n and sorted_rows[pos][1] == sorted_rows[run_start][1]:
                        pos += 1
                    for j in range(run_start, pos):
                        result_values[sorted_rows[j][0]] = dense
                    dense += 1

            elif rank_type == "percent_rank":
                # First compute standard rank
                ranks: list[int] = [0] * n
                pos = 0
                while pos < n:
                    run_start = pos
                    while pos < n and sorted_rows[pos][1] == sorted_rows[run_start][1]:
                        pos += 1
                    rank_val = run_start + 1
                    for j in range(run_start, pos):
                        ranks[j] = rank_val
                # percent_rank = (rank - 1) / (n - 1), or 0.0 if n == 1
                for j in range(n):
                    idx = sorted_rows[j][0]
                    if n == 1:
                        result_values[idx] = 0.0
                    else:
                        result_values[idx] = (ranks[j] - 1) / (n - 1)

            elif rank_type.startswith("ntile_"):
                ntile_n = int(rank_type[len("ntile_") :])
                for pos, (idx, _) in enumerate(sorted_rows):
                    # Standard ntile: bucket = ceil((pos+1) * ntile_n / n)
                    bucket = (pos * ntile_n) // n + 1
                    result_values[idx] = bucket

            elif rank_type.startswith("top_"):
                top_n = int(rank_type[len("top_") :])
                # Sort DESC for top-N with nulls last
                desc_rows = sorted(
                    sorted_rows,
                    key=lambda x: (x[1] is None, -(x[1]) if x[1] is not None else 0),
                )
                for pos, (idx, _) in enumerate(desc_rows):
                    result_values[idx] = pos + 1 <= top_n

            elif rank_type.startswith("bottom_"):
                bottom_n = int(rank_type[len("bottom_") :])
                # Already sorted ASC for bottom-N
                for pos, (idx, _) in enumerate(sorted_rows):
                    result_values[idx] = pos + 1 <= bottom_n

            else:
                raise ValueError(f"Unsupported rank type: {rank_type}")

        new_col = pa.array(result_values)
        return table.append_column(feature_name, new_col)
