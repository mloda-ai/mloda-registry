"""PythonDict implementation of gap-threshold sessionization.

Unlike ffill/EMA (state resets per partition), a sessionization session id is
a globally-unique, 0-based cumulative count over the whole sorted frame: rows
are sorted by ``[*partition_by, order_by]`` ascending, a row starts a new
session when it is first-in-partition or the gap to the previous row exceeds
the threshold, and the running session counter itself is never reset between
partitions, only the first-in-partition flag resets. Results are then
scattered back to original row position.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_utils import row_count

from mloda.community.feature_groups.data_operations.python_dict_helpers import (
    is_nan,
    nulls_last_sort_key,
    partition_sort_key,
    values_equal,
)
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

        def sort_key(i: int) -> tuple[Any, ...]:
            # Partition-key components use partition_sort_key so a None-keyed partition and a
            # NaN-keyed partition stay distinct, contiguous groups (order_by ties would
            # otherwise interleave them if both used nulls_last_sort_key's shared null tier).
            return tuple(partition_sort_key(v) for v in partition_key(i)) + (nulls_last_sort_key(order_vals[i]),)

        # Sort row indices by [*partition_by, order_by] ascending (stable), nulls last.
        sorted_indices = sorted(range(num_rows), key=sort_key)

        threshold = timedelta(seconds=threshold_seconds)
        result_values: list[Any] = [None] * num_rows

        session_id = -1
        poisoned = False

        for pos, row_index in enumerate(sorted_indices):
            key = partition_key(row_index)
            order_val = order_vals[row_index]

            if pos == 0:
                ambiguous = False
                is_new = True
            else:
                prev_index = sorted_indices[pos - 1]
                prev_key = partition_key(prev_index)
                prev_order_val = order_vals[prev_index]
                # A null on either side of the partition-key or order-by comparison makes
                # "same partition?"/"gap exceeded?" unknown rather than False. That mirrors
                # the PyArrow oracle's Arrow-compute-kernel comparisons (``not_equal``,
                # ``greater``), which yield null whenever an operand is null; once such a
                # null reaches ``pc.cumulative_sum`` (default ``skip_nulls=False``), the
                # session id for that row and every following row in sorted order becomes
                # (and stays) null, rather than resuming the count once nulls are behind.
                # A NaN order_by value is ambiguous too (the gap subtraction below is not
                # well-defined against it), but a NaN partition-key value is NOT: it groups
                # with itself via ``values_equal`` below, matching PyArrow's group_by().
                ambiguous = (
                    any(v is None for v in key)
                    or any(v is None for v in prev_key)
                    or order_val is None
                    or prev_order_val is None
                    or is_nan(order_val)
                    or is_nan(prev_order_val)
                )
                is_new = not ambiguous and (not values_equal(key, prev_key) or (order_val - prev_order_val) > threshold)

            if ambiguous:
                poisoned = True
            elif is_new:
                session_id += 1

            result_values[row_index] = None if poisoned else session_id

        result[feature_name] = result_values
        return result
