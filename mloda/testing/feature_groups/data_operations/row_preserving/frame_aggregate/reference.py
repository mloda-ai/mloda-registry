"""Test reference implementation for frame aggregate feature groups.

Accepts PyArrow tables but computes in Python. Used as the cross-framework
comparison baseline in test suites. Uses bulk ``to_pylist()`` extraction
(one C++ call per column) instead of per-row ``.as_py()`` calls, then
performs grouping, sorting, and windowed aggregation in pure Python on the
extracted lists.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.testing.feature_groups.data_operations.aggregation_helpers import aggregate
from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.base import (
    FrameAggregateFeatureGroup,
)


class ReferenceFrameAggregate(FrameAggregateFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
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
        frame_size: int | None = None,
        frame_unit: str | None = None,
    ) -> pa.Table:
        num_rows = table.num_rows

        # Bulk-extract all needed columns (one C++ call each, not N .as_py() calls)
        partition_lists = {col: table.column(col).to_pylist() for col in partition_by}
        order_vals = table.column(order_by).to_pylist()
        source_vals = table.column(source_col).to_pylist()

        # Build group keys from extracted Python lists (no PyArrow calls in loop)
        groups: dict[tuple[Any, ...], list[tuple[int, Any, Any]]] = {}
        for i in range(num_rows):
            key = tuple(partition_lists[col][i] for col in partition_by)
            if key not in groups:
                groups[key] = []
            groups[key].append((i, order_vals[i], source_vals[i]))

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

        Uses timedelta for second, minute, hour, day, and week units, and
        relativedelta for month and year units to ensure calendar-accurate windows.
        """
        from datetime import timedelta

        from dateutil.relativedelta import relativedelta

        unit_map = {
            "second": timedelta(seconds=size),
            "minute": timedelta(minutes=size),
            "hour": timedelta(hours=size),
            "day": timedelta(days=size),
            "week": timedelta(weeks=size),
            "month": relativedelta(months=size),
            "year": relativedelta(years=size),
        }
        delta = unit_map.get(unit, timedelta(days=size))

        if current_order is None:
            return [rows[pos][2]]

        window_start = current_order - delta
        return [r[2] for r in rows[: pos + 1] if r[1] is not None and r[1] >= window_start]
