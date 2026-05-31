"""PyArrow implementation of resample (production AND reference oracle).

The cross-framework spec is pinned to this implementation: epoch-anchored
``pyarrow.compute.floor_temporal`` floor, then ``Table.group_by`` /
``aggregate``. PyArrow's ``sum`` / ``mean`` over an all-null bucket return
``None`` and ``count`` returns the non-null count (0) -- that is the behaviour
every other backend must match.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_changing.resample.base import (
    RESAMPLE_AGGS,
    ResampleFeatureGroup,
)

# Resample agg -> PyArrow group_by aggregate func name.
_PA_AGG_FUNCS: dict[str, str] = {
    "mean": "mean",
    "sum": "sum",
    "count": "count",
    "min": "min",
    "max": "max",
}


class PyArrowResample(ResampleFeatureGroup):
    """PyArrow backend; also the cross-framework reference implementation."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PyArrowTable}

    @classmethod
    def _assert_time_column_present(cls, data: pa.Table, time_column: str) -> None:
        if time_column not in data.schema.names:
            raise ValueError(
                f"time_column {time_column!r} is not present in the PyArrow table; available: {data.schema.names}."
            )

    @classmethod
    def _assert_source_column_present(cls, data: pa.Table, source_col: str) -> None:
        if source_col not in data.schema.names:
            raise ValueError(
                f"Source column {source_col!r} is not present in the PyArrow table; available: {data.schema.names}."
            )

    @classmethod
    def _compute_resample(
        cls,
        data: pa.Table,
        feature_name: str,
        source_col: str,
        time_column: str,
        partition_by: list[str],
        n: int,
        unit: str,
        agg: str,
    ) -> pa.Table:
        if agg not in _PA_AGG_FUNCS:
            raise ValueError(f"Unsupported resample agg {agg!r} for PyArrow; supported: {sorted(RESAMPLE_AGGS)}.")

        # Floor the time column in place (bucket start keeps the original name).
        bucket = pc.floor_temporal(data.column(time_column), multiple=n, unit=unit)
        data = data.set_column(data.schema.get_field_index(time_column), time_column, bucket)

        keys = [*partition_by, time_column]
        pa_func = _PA_AGG_FUNCS[agg]
        grouped = data.group_by(keys).aggregate([(source_col, pa_func)])

        # Rename the auto-generated aggregate column to the feature name.
        auto_col = f"{source_col}_{pa_func}"
        names = [feature_name if c == auto_col else c for c in grouped.column_names]
        return grouped.rename_columns(names)
