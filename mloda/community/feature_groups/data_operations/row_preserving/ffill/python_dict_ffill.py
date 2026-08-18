"""PythonDict implementation of ffill-by-time.

Builds per-partition-group row lists (stable-sorted by ``order_by``, nulls
last), carries the last non-null source value forward through each sorted
group, then scatters results back to original row position.
"""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils import row_count

from mloda.community.feature_groups.data_operations.row_preserving.ffill.base import FfillFeatureGroup


class PythonDictFfill(FfillFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def _assert_source_column_present(cls, data: dict[str, list[Any]], source_col: str) -> None:
        if source_col not in data:
            raise ValueError(
                f"Source column {source_col!r} is not present in the PythonDict data; available: {list(data.keys())}."
            )

    @classmethod
    def _compute_ffill(
        cls,
        data: dict[str, list[Any]],
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        order_by: str,
    ) -> dict[str, list[Any]]:
        num_rows = row_count(data)

        order_vals = data[order_by]
        source_vals = data[source_col]
        partition_cols = [data[col] for col in partition_by]

        # Build group keys, then stable-sort each group by order_by (nulls last).
        groups: dict[tuple[Any, ...], list[tuple[int, Any, Any]]] = {}
        for i in range(num_rows):
            key = tuple(col[i] for col in partition_cols)
            groups.setdefault(key, []).append((i, order_vals[i], source_vals[i]))

        for rows in groups.values():
            rows.sort(key=lambda t: (t[1] is None, t[1] if t[1] is not None else 0))

        result_values: list[Any] = [None] * num_rows

        for rows in groups.values():
            carried: Any = None
            for row_index, _order_val, value in rows:
                if value is not None:
                    carried = value
                result_values[row_index] = carried

        result = dict(data)
        result[feature_name] = result_values
        return result
