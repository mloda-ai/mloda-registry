"""PyArrow implementation of resample (production AND reference oracle).

The cross-framework spec is pinned to this implementation: epoch-anchored
``pyarrow.compute.floor_temporal`` floor, then ``Table.group_by`` /
``aggregate``. PyArrow's ``sum`` / ``mean`` over an all-null bucket return
``None`` and ``count`` returns the non-null count (0) -- that is the behaviour
every other backend must match.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.pyarrow_helpers import rename_aggregates_by_position
from mloda.community.feature_groups.data_operations.row_changing.resample.base import (
    RESAMPLE_AGGS,
    ResampleAgg,
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


class PyArrowResample(Implementation[ResampleFeatureGroup]):
    """PyArrow backend; also the cross-framework reference implementation."""

    feature_group = ResampleFeatureGroup
    compute_framework = PyArrowTable

    @classmethod
    def calculate(
        cls, group: type[ResampleFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.resample(data, features, compute_resample=cls._compute_resample)

    @classmethod
    def _compute_resample(
        cls,
        data: pa.Table,
        aggs: list[ResampleAgg],
        time_column: str,
        partition_by: list[str],
        n: int,
        unit: str,
    ) -> pa.Table:
        for spec in aggs:
            if spec.agg not in _PA_AGG_FUNCS:
                raise ValueError(
                    f"Unsupported resample agg {spec.agg!r} for PyArrow; supported: {sorted(RESAMPLE_AGGS)}."
                )

        # Floor the time column in place (bucket start keeps the original name).
        bucket = pc.floor_temporal(data.column(time_column), multiple=n, unit=unit)
        data = data.set_column(data.schema.get_field_index(time_column), time_column, bucket)

        keys = [*partition_by, time_column]
        grouped = data.group_by(keys).aggregate([(spec.source_col, _PA_AGG_FUNCS[spec.agg]) for spec in aggs])

        return rename_aggregates_by_position(grouped, keys, [spec.feature_name for spec in aggs])
