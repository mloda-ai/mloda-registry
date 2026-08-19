"""PythonDict implementation for offset feature groups.

Builds per-partition-group row lists (stable-sorted by ``order_by``, nulls
last), computes the requested offset value from each group's value list,
then scatters results back to original row position.
"""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils import row_count

from mloda.community.feature_groups.data_operations.python_dict_helpers import group_key_value, nulls_last_sort_key
from mloda.community.feature_groups.data_operations.row_preserving.offset.base import (
    OffsetFeatureGroup,
)


class PythonDictOffset(OffsetFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def _compute_offset(
        cls,
        data: dict[str, list[Any]],
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        order_by: str,
        offset_type: str,
    ) -> dict[str, list[Any]]:
        num_rows = row_count(data)

        order_vals = data[order_by]
        source_vals = data[source_col]
        partition_cols = [data[col] for col in partition_by]

        # Build group keys, then stable-sort each group by order_by (nulls last).
        groups: dict[tuple[Any, ...], list[tuple[int, Any, Any]]] = {}
        for i in range(num_rows):
            key = tuple(group_key_value(col[i]) for col in partition_cols)
            groups.setdefault(key, []).append((i, order_vals[i], source_vals[i]))

        for rows in groups.values():
            rows.sort(key=lambda t: nulls_last_sort_key(t[1]))

        result_values: list[Any] = [None] * num_rows

        for rows in groups.values():
            cls._apply_offset(rows, offset_type, result_values)

        result = dict(data)
        result[feature_name] = result_values
        return result

    @classmethod
    def _apply_offset(
        cls,
        sorted_rows: list[tuple[int, Any, Any]],
        offset_type: str,
        result_values: list[Any],
    ) -> None:
        """Compute one offset type for a single (already order-sorted) partition group."""
        n = len(sorted_rows)
        vals = [row[2] for row in sorted_rows]

        if offset_type.startswith("lag_"):
            offset_n = int(offset_type[len("lag_") :])
            for pos in range(n):
                if pos >= offset_n:
                    result_values[sorted_rows[pos][0]] = vals[pos - offset_n]

        elif offset_type.startswith("lead_"):
            offset_n = int(offset_type[len("lead_") :])
            for pos in range(n):
                if pos + offset_n < n:
                    result_values[sorted_rows[pos][0]] = vals[pos + offset_n]

        elif offset_type.startswith("diff_"):
            offset_n = int(offset_type[len("diff_") :])
            for pos in range(n):
                curr = vals[pos]
                if pos >= offset_n and curr is not None and vals[pos - offset_n] is not None:
                    result_values[sorted_rows[pos][0]] = curr - vals[pos - offset_n]

        elif offset_type.startswith("pct_change_"):
            offset_n = int(offset_type[len("pct_change_") :])
            for pos in range(n):
                curr = vals[pos]
                prev = vals[pos - offset_n] if pos >= offset_n else None
                if curr is not None and prev is not None and prev != 0:
                    result_values[sorted_rows[pos][0]] = (curr - prev) / prev

        elif offset_type == "first_value":
            first = next((v for v in vals if v is not None), None)
            for pos in range(n):
                result_values[sorted_rows[pos][0]] = first

        elif offset_type == "last_value":
            last = next((v for v in reversed(vals) if v is not None), None)
            for pos in range(n):
                result_values[sorted_rows[pos][0]] = last

        else:
            raise ValueError(f"Unsupported offset type: {offset_type}")
