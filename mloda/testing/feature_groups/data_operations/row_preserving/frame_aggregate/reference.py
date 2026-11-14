"""Pure Python reference for frame aggregate: runs the loop of the group passed in with a kernel-backed compute.

pa.Table is only the I/O container; grouping, ordering and windowing run on ``to_pylist()`` lists.
"""

from __future__ import annotations

import calendar
from datetime import timedelta
from typing import Any

import pyarrow as pa
from mloda.provider import FeatureSet

from mloda.testing.feature_groups.data_operations.aggregation_helpers import (
    SUPPORTED_AGG_TYPES,
    aggregate,
    declared_type,
    group_rows,
    order_key,
    unsupported_agg_type_error,
)
from mloda.testing.feature_groups.data_operations.helpers import masked_column


def _compute_frame(
    table: pa.Table,
    feature_name: str,
    source_col: str,
    partition_by: list[str],
    order_by: str,
    agg_type: str,
    frame_type: str,
    frame_size: int | None = None,
    frame_unit: str | None = None,
    mask_spec: list[tuple[str, str, Any]] | None = None,
) -> pa.Table:
    if agg_type not in SUPPORTED_AGG_TYPES:
        raise unsupported_agg_type_error(agg_type, SUPPORTED_AGG_TYPES, framework="Reference")

    out_type = declared_type(agg_type, table.schema.field(source_col).type)
    num_rows = table.num_rows
    keys = [table.column(col).to_pylist() for col in partition_by]
    order_vals = table.column(order_by).to_pylist()  # unmasked: the mask must not null the order key
    source = masked_column(table, source_col, mask_spec) if mask_spec is not None else table.column(source_col)
    source_vals = source.to_pylist()

    result_values: list[Any] = [None] * num_rows
    for indices in group_rows(keys, num_rows).values():
        rows = [(i, order_vals[i], source_vals[i]) for i in sorted(indices, key=lambda i: order_key(order_vals[i]))]
        for pos, (orig_idx, order_val, _) in enumerate(rows):
            if frame_type == "rolling":
                wsize = int(frame_size) if frame_size is not None else 1
                window = [r[2] for r in rows[max(0, pos - wsize + 1) : pos + 1]]
            elif frame_type == "time":
                window = _time_window(rows, pos, order_val, frame_size or 1, str(frame_unit or "day"))
            elif frame_type in ("cumulative", "expanding"):
                window = [r[2] for r in rows[: pos + 1]]
            else:
                raise ValueError(f"Unknown frame_type '{frame_type}'")

            result_values[orig_idx] = aggregate(window, agg_type)

    return table.append_column(feature_name, pa.array(result_values, type=out_type))


def _subtract_calendar_months(value: Any, months: int) -> Any:
    """``value`` minus whole calendar months, the day clamped to the target month's length."""
    index = value.year * 12 + (value.month - 1) - months
    year, month = divmod(index, 12)
    month += 1
    return value.replace(year=year, month=month, day=min(value.day, calendar.monthrange(year, month)[1]))


def _time_window(
    rows: list[tuple[int, Any, Any]],
    pos: int,
    current_order: Any,
    size: int,
    unit: str,
) -> list[Any]:
    """Collect values within a time-based window ending at the current row.

    Second, minute, hour, day and week subtract a timedelta; month and year subtract calendar
    months (a year is 12), clamping the day to the target month's length.
    """
    if current_order is None:
        return [rows[pos][2]]

    if unit in ("month", "year"):
        window_start = _subtract_calendar_months(current_order, size * (12 if unit == "year" else 1))
    else:
        deltas = {
            "second": timedelta(seconds=size),
            "minute": timedelta(minutes=size),
            "hour": timedelta(hours=size),
            "week": timedelta(weeks=size),
        }
        window_start = current_order - deltas.get(unit, timedelta(days=size))
    return [r[2] for r in rows[: pos + 1] if r[1] is not None and r[1] >= window_start]


def reference_frame_aggregate(group: Any, table: pa.Table, features: FeatureSet) -> pa.Table:
    """Run the group's frame aggregate loop with the pure-Python reference compute."""
    return group.aggregate(table, features, compute_frame=_compute_frame)
