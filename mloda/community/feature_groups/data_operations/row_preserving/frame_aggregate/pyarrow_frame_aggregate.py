"""PyArrow implementation for frame aggregate feature groups."""

from __future__ import annotations

from typing import Any, Optional, Set, Type, Union

import pyarrow as pa

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.pyarrow_aggregation_helpers import aggregate
from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.base import (
    FrameAggregateFeatureGroup,
)


class PyArrowFrameAggregate(FrameAggregateFeatureGroup):
    """Pure-Python reference implementation of frame aggregation over PyArrow tables.

    This implementation iterates row-by-row and is O(n * w) per partition (where
    n is partition size and w is window size). It is designed for correctness
    testing and small datasets. For production workloads with large tables,
    prefer the Pandas, Polars, or DuckDB backends which use vectorized operations.
    """

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PyArrowTable}

    @classmethod
    def _compute_frame(
        cls,
        table: pa.Table,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        order_by: str,
        agg_type: str,
        frame_type: str,
        frame_size: Optional[int] = None,
        frame_unit: Optional[str] = None,
    ) -> pa.Table:
        num_rows = table.num_rows

        keys: list[tuple[Any, ...]] = []
        for i in range(num_rows):
            key = tuple(table.column(col)[i].as_py() for col in partition_by)
            keys.append(key)

        groups: dict[tuple[Any, ...], list[tuple[int, Any, Any]]] = {}
        for i in range(num_rows):
            key = keys[i]
            val = table.column(source_col)[i].as_py()
            order_val = table.column(order_by)[i].as_py()
            if key not in groups:
                groups[key] = []
            groups[key].append((i, order_val, val))

        for key in groups:
            groups[key].sort(key=lambda t: (t[1] is None, t[1] if t[1] is not None else 0))

        result_values: list[Any] = [None] * num_rows

        for key, rows in groups.items():
            for pos, (orig_idx, order_val, val) in enumerate(rows):
                if frame_type == "rolling":
                    wsize = int(frame_size) if frame_size is not None else 1
                    window_start = max(0, pos - wsize + 1)
                    window = [r[2] for r in rows[window_start : pos + 1]]
                elif frame_type in ("cumulative", "expanding"):
                    window = [r[2] for r in rows[: pos + 1]]
                elif frame_type == "time":
                    window = cls._time_window(rows, pos, order_val, frame_size or 1, str(frame_unit or "day"))
                else:
                    window = [r[2] for r in rows[: pos + 1]]

                result_values[orig_idx] = aggregate(window, agg_type)

        new_col = pa.array(result_values)
        return table.append_column(feature_name, new_col)

    @classmethod
    def _time_window(
        cls,
        rows: list[tuple[int, Any, Any]],
        pos: int,
        current_order: Any,
        size: int,
        unit: str,
    ) -> list[Any]:
        """Collect values within a time-based window ending at the current row.

        Note: month and year use fixed approximations (30 days per month,
        365 days per year). For calendar-accurate windows, consider using
        dateutil.relativedelta in a custom subclass.
        """
        from datetime import timedelta

        unit_map = {
            "second": timedelta(seconds=size),
            "minute": timedelta(minutes=size),
            "hour": timedelta(hours=size),
            "day": timedelta(days=size),
            "week": timedelta(weeks=size),
            "month": timedelta(days=size * 30),
            "year": timedelta(days=size * 365),
        }
        delta = unit_map.get(unit, timedelta(days=size))

        if current_order is None:
            return [rows[pos][2]]

        window_start = current_order - delta
        return [r[2] for r in rows[: pos + 1] if r[1] is not None and r[1] >= window_start]
