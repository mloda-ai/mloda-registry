"""PyArrow implementation of resample (a first backend).

Epoch-anchored ``pyarrow.compute.floor_temporal`` floor, then ``Table.group_by`` /
``aggregate``. PyArrow's ``sum`` / ``mean`` over an all-null bucket return
``None`` and ``count`` returns the non-null count (0), which matches the reference.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.helper_columns import unique_helper_name
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
    """PyArrow backend."""

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
        names = [spec.feature_name for spec in aggs]
        taken = set(data.column_names) | set(names)
        nan_helpers: dict[str, str] = {}
        for spec in aggs:
            if (
                spec.agg in {"min", "max"}
                and pa.types.is_floating(data.column(spec.source_col).type)
                and spec.source_col not in nan_helpers
            ):
                helper = unique_helper_name("__mloda_resample_nan__", taken)
                taken.add(helper)
                nan_helpers[spec.source_col] = helper
                data = data.append_column(helper, pc.is_nan(data.column(spec.source_col)))

        entries = [(spec.source_col, _PA_AGG_FUNCS[spec.agg]) for spec in aggs]
        entries.extend((helper, "all") for helper in nan_helpers.values())
        grouped = data.group_by(keys).aggregate(entries)
        result = rename_aggregates_by_position(grouped, keys, [*names, *nan_helpers.values()])

        # Older Arrow returns +/-inf for all-NaN extrema; repair only those buckets.
        for spec in aggs:
            if spec.agg in {"min", "max"} and spec.source_col in nan_helpers:
                values = result.column(spec.feature_name)
                all_nan = pc.fill_null(result.column(nan_helpers[spec.source_col]), False)
                corrected = pc.if_else(all_nan, pa.scalar(float("nan"), type=values.type), values)
                result = result.set_column(
                    result.schema.get_field_index(spec.feature_name), spec.feature_name, corrected
                )

        return result.drop_columns(list(nan_helpers.values()))
