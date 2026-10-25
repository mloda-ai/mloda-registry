"""Pure Python reference for frame aggregate: the shared loop with a kernel-backed compute.

pa.Table is only the I/O container; grouping, ordering and windowing run on ``to_pylist()`` lists.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
from mloda.provider import FeatureSet

from mloda.community.feature_groups.data_operations.mask_utils import apply_pyarrow_mask
from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.base import (
    FrameAggregateFeatureGroup,
)
from mloda.testing.feature_groups.data_operations.aggregation_helpers import (
    aggregate,
    declared_type,
    group_rows,
    order_key,
)


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
    num_rows = table.num_rows
    keys = [table.column(col).to_pylist() for col in partition_by]
    order_vals = table.column(order_by).to_pylist()  # unmasked: the mask must not null the order key
    source = apply_pyarrow_mask(table, source_col, mask_spec) if mask_spec is not None else table.column(source_col)
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
            else:
                window = [r[2] for r in rows[: pos + 1]]

            result_values[orig_idx] = aggregate(window, agg_type)

    out_type = declared_type(agg_type, table.schema.field(source_col).type)
    return table.append_column(feature_name, pa.array(result_values, type=out_type))


def _time_window(
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


def reference_frame_aggregate(table: pa.Table, features: FeatureSet) -> pa.Table:
    """Run the shared frame aggregate loop with the pure-Python reference compute."""
    return FrameAggregateFeatureGroup.aggregate(table, features, compute_frame=_compute_frame)
