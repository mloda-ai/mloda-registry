"""PythonDict implementation for rank feature groups.

Builds per-partition groups of ``(row index, order_by value)``, stable-sorts
each group ascending with nulls last, and computes ranks from position in
that sorted order: a null ``order_by`` row naturally lands at the highest
rank number instead of receiving a null rank. Results are then scattered
back to original row position.
"""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils import row_count

from mloda.community.feature_groups.data_operations.row_preserving.rank.base import (
    RankFeatureGroup,
)


class PythonDictRank(RankFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def _compute_rank(
        cls,
        data: dict[str, list[Any]],
        feature_name: str,
        partition_by: list[str],
        order_by: str,
        rank_type: str,
    ) -> dict[str, list[Any]]:
        num_rows = row_count(data)

        order_vals = data[order_by]
        partition_cols = [data[col] for col in partition_by]

        # Build group keys, then stable-sort each group by order_by (nulls last).
        groups: dict[tuple[Any, ...], list[tuple[int, Any]]] = {}
        for i in range(num_rows):
            key = tuple(col[i] for col in partition_cols)
            groups.setdefault(key, []).append((i, order_vals[i]))

        for rows in groups.values():
            rows.sort(key=lambda x: (1,) if x[1] is None else (0, x[1]))

        result_values: list[Any] = [None] * num_rows

        for sorted_rows in groups.values():
            cls._apply_rank(sorted_rows, rank_type, result_values)

        result = dict(data)
        result[feature_name] = result_values
        return result

    @classmethod
    def _apply_rank(
        cls,
        sorted_rows: list[tuple[int, Any]],
        rank_type: str,
        result_values: list[Any],
    ) -> None:
        """Compute one rank type for a single (already order-sorted, nulls-last) partition group."""
        n = len(sorted_rows)

        if rank_type == "row_number":
            for pos, (idx, _) in enumerate(sorted_rows):
                result_values[idx] = pos + 1

        elif rank_type == "rank":
            pos = 0
            while pos < n:
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
            # Reverse the ASC-sorted non-null rows to get DESC; keep nulls last
            non_null = [(idx, val) for idx, val in sorted_rows if val is not None]
            nulls = [(idx, val) for idx, val in sorted_rows if val is None]
            desc_rows = non_null[::-1] + nulls
            for pos, (idx, _) in enumerate(desc_rows):
                result_values[idx] = pos + 1 <= top_n

        elif rank_type.startswith("bottom_"):
            bottom_n = int(rank_type[len("bottom_") :])
            # Already sorted ASC for bottom-N
            for pos, (idx, _) in enumerate(sorted_rows):
                result_values[idx] = pos + 1 <= bottom_n

        else:
            raise ValueError(f"Unsupported rank type: {rank_type}")
