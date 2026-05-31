"""PyArrow implementation of gap-threshold sessionization (production AND reference).

PyArrow is the cross-framework reference oracle for sessionization. The
computation is fully native (Arrow compute kernels, no per-row Python loop):
rows are sorted by ``[*partition_by, order_col]`` ascending, the gap to the
previous row and the partition boundary are derived with shifted-slice
comparisons, the per-row ``is_new`` flag is cumulatively summed
(``pc.cumulative_sum``) to produce the 0-based session id, then the result is
scattered back to the original row order.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.sessionization.base import (
    SessionizationFeatureGroup,
)


class PyArrowSessionization(SessionizationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PyArrowTable}

    @classmethod
    def _assert_source_column_present(cls, data: pa.Table, order_col: str) -> None:
        if order_col not in data.column_names:
            raise ValueError(
                f"Source column {order_col!r} is not present in the PyArrow table; available: {data.column_names}."
            )

    @classmethod
    def _compute_session(
        cls,
        data: pa.Table,
        feature_name: str,
        order_col: str,
        threshold_seconds: int,
        partition_by: list[str],
    ) -> pa.Table:
        order_type = data.column(order_col).type
        if not pa.types.is_timestamp(order_type):
            raise ValueError(f"Order column {order_col!r} must be a timestamp for sessionization; got {order_type}.")

        n = data.num_rows
        if n == 0:
            return data.append_column(feature_name, pa.array([], type=pa.int64()))

        sort_keys = [(col, "ascending") for col in (*partition_by, order_col)]
        perm = pc.sort_indices(data, sort_keys=sort_keys)
        sorted_tbl = data.take(perm)

        # Normalize to microsecond resolution before the int64 view so any input
        # resolution (s/ms/us/ns) yields microseconds since epoch.
        col = sorted_tbl.column(order_col)
        col_us = pc.cast(col, pa.timestamp("us", tz=col.type.tz))
        ts_int = pc.cast(col_us, pa.int64())
        threshold_us = threshold_seconds * 1_000_000

        if n == 1:
            is_new = pa.array([True], type=pa.bool_())
        else:
            diff = pc.subtract(ts_int.slice(1), ts_int.slice(0, n - 1))
            gap_new = pc.greater(diff, pa.scalar(threshold_us, type=pa.int64()))

            part_changed: pa.Array = pa.array([False] * (n - 1), type=pa.bool_())
            for col in partition_by:
                sorted_col = sorted_tbl.column(col)
                changed = pc.not_equal(sorted_col.slice(1), sorted_col.slice(0, n - 1))
                part_changed = pc.or_(part_changed, changed)

            tail = pc.or_(gap_new, part_changed)
            tail_arr = pc.cast(tail, pa.bool_())
            if isinstance(tail_arr, pa.ChunkedArray):
                tail_arr = tail_arr.combine_chunks()
            is_new = pa.concat_arrays([pa.array([True], type=pa.bool_()), tail_arr])

        sessions_sorted = pc.subtract(
            pc.cumulative_sum(pc.cast(is_new, pa.int64())),
            pa.scalar(1, type=pa.int64()),
        )

        # Scatter the session ids back to the original row order.
        inv = pc.sort_indices(perm)
        sessions = pc.take(sessions_sorted, inv)

        return data.append_column(feature_name, pc.cast(sessions, pa.int64()))
