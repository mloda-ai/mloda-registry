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

from mloda.provider import ComputeFramework, FeatureSet, Implementation
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


class PythonDictSessionization(Implementation[SessionizationFeatureGroup]):
    feature_group = SessionizationFeatureGroup
    compute_framework = PythonDictFramework

    @classmethod
    def calculate(
        cls,
        group: type[SessionizationFeatureGroup],
        data: Any,
        features: FeatureSet,
        framework: ComputeFramework | None,
    ) -> Any:
        return group.sessionize(data, features, compute_session=cls._compute_session)

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

        for pos, row_index in enumerate(sorted_indices):
            order_val = order_vals[row_index]
            if pos == 0 or order_val is None or is_nan(order_val):
                # A null timestamp never ties, so it starts its own session.
                is_new = True
            else:
                prev_index = sorted_indices[pos - 1]
                prev_order_val = order_vals[prev_index]
                is_new = not values_equal(partition_key(row_index), partition_key(prev_index)) or (
                    prev_order_val is None or is_nan(prev_order_val) or (order_val - prev_order_val) > threshold
                )

            if is_new:
                session_id += 1
            result_values[row_index] = session_id

        result[feature_name] = result_values
        return result
