"""PythonDict implementation of gap-threshold sessionization.

Pure-Python, dependency-free implementation with no engine-level windowing
limitations, so it targets FULL support: all five existing backends compute
sessionization natively, and PythonDict is no exception.

Unlike ffill/EMA (state resets per partition), a sessionization session id is a
GLOBALLY-UNIQUE 0-based cumulative count over the WHOLE sorted frame: rows are
sorted by ``[*partition_by, order_by]`` ascending (``partition_by`` groups
same-partition rows together so "first-in-partition" can be detected), a row
starts a new session when it is first-in-partition OR the gap to the previous
row (in the sorted sequence) is strictly greater than the threshold, and the
running session counter itself is NEVER reset between partitions -- only the
"is this the first row I've seen in this partition" flag resets. Results are
then scattered back to a list indexed by original row position so the output
row order matches the input (row-preserving).

PythonDict sessionizes native ``datetime`` values directly: the gap is a plain
``timedelta`` subtraction compared against
``timedelta(seconds=threshold_seconds)``, with no timestamp-resolution casting
concerns (unlike PyArrow, which must normalize mixed-resolution timestamp
types to a common unit before taking an int64 view).
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils import row_count

from mloda.community.feature_groups.data_operations.row_preserving.sessionization.base import (
    SessionizationFeatureGroup,
)


class PythonDictSessionization(SessionizationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def _assert_source_column_present(cls, data: dict[str, list[Any]], order_col: str) -> None:
        if order_col not in data:
            raise ValueError(
                f"Source column {order_col!r} is not present in the PythonDict data; available: {list(data.keys())}."
            )

    @classmethod
    def _compute_session(
        cls,
        data: dict[str, list[Any]],
        feature_name: str,
        order_col: str,
        threshold_seconds: int,
        partition_by: list[str],
    ) -> dict[str, list[Any]]:
        num_rows = row_count(data)

        result = dict(data)
        if num_rows == 0:
            result[feature_name] = []
            return result

        order_vals = data[order_col]
        partition_cols = [data[col] for col in partition_by]

        def partition_key(i: int) -> tuple[Any, ...]:
            return tuple(col[i] for col in partition_cols)

        # Sort row indices by [*partition_by, order_by] ascending (stable), so
        # same-partition rows are grouped together and time-ordered within
        # each group. The session-id cumsum below runs over this single
        # global sequence in one pass; only the first-in-partition flag
        # resets per group, the running session counter never does.
        sorted_indices = sorted(range(num_rows), key=lambda i: (partition_key(i), order_vals[i]))

        threshold = timedelta(seconds=threshold_seconds)
        result_values: list[Any] = [None] * num_rows

        session_id = -1
        prev_key: tuple[Any, ...] | None = None
        prev_order_val: Any = None

        for row_index in sorted_indices:
            key = partition_key(row_index)
            order_val = order_vals[row_index]

            is_new = prev_key is None or key != prev_key or (order_val - prev_order_val) > threshold
            if is_new:
                session_id += 1

            result_values[row_index] = session_id
            prev_key = key
            prev_order_val = order_val

        result[feature_name] = result_values
        return result
