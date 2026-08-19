"""PythonDict implementation of resample.

Floors each row's ``time_column`` to its ``(n, unit)`` bucket (shares
``floor_fixed_duration`` with ``python_dict_time_bucketization``; see that
module for the epoch-anchor rationale), groups row indices by
``(*partition_by, bucket_start)``, then reduces each group's ``source_col``
values with the requested aggregation, skipping nulls. A non-empty bucket
with no non-null values still emits: ``count = 0``, other aggs ``None``.
"""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error
from mloda.community.feature_groups.data_operations.python_dict_helpers import (
    SECONDS_PER_UNIT,
    floor_fixed_duration,
    group_key_value,
    reduce_agg,
)
from mloda.community.feature_groups.data_operations.row_changing.resample.base import (
    RESAMPLE_AGGS,
    ResampleFeatureGroup,
)


def _floor_dt(dt: Any, n: int, unit: str) -> Any:
    """Floor ``dt`` to the start of its ``(n, unit)`` bucket, preserving tzinfo."""
    if unit in SECONDS_PER_UNIT:
        return floor_fixed_duration(dt, n, unit)
    raise ValueError(f"Unsupported resample unit for PythonDict: {unit!r}")


def _reduce(agg: str, values: list[Any]) -> Any:
    """Reduce one bucket's raw (possibly null-containing) values to a single result."""
    if agg not in RESAMPLE_AGGS:
        raise unsupported_agg_type_error(agg, RESAMPLE_AGGS, framework="PythonDict")
    return reduce_agg(agg, values)


class PythonDictResample(ResampleFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def _assert_time_column_present(cls, data: dict[str, list[Any]], time_column: str) -> None:
        if time_column not in data:
            raise ValueError(
                f"time_column {time_column!r} is not present in the PythonDict data; available: {list(data)}."
            )

    @classmethod
    def _assert_source_column_present(cls, data: dict[str, list[Any]], source_col: str) -> None:
        if source_col not in data:
            raise ValueError(
                f"Source column {source_col!r} is not present in the PythonDict data; available: {list(data)}."
            )

    @classmethod
    def _compute_resample(
        cls,
        data: dict[str, list[Any]],
        feature_name: str,
        source_col: str,
        time_column: str,
        partition_by: list[str],
        n: int,
        unit: str,
        agg: str,
    ) -> dict[str, list[Any]]:
        if agg not in RESAMPLE_AGGS:
            raise unsupported_agg_type_error(agg, RESAMPLE_AGGS, framework="PythonDict")

        time_col = data[time_column]
        source_values = data[source_col]
        partition_cols = [data[col] for col in partition_by]

        buckets = [None if value is None else _floor_dt(value, n, unit) for value in time_col]

        # Group row indices by (*partition_by, bucket_start), first-occurrence order.
        groups: dict[tuple[Any, ...], list[int]] = {}
        for i in range(len(time_col)):
            key = tuple(group_key_value(col[i]) for col in partition_cols) + (buckets[i],)
            groups.setdefault(key, []).append(i)

        result: dict[str, list[Any]] = {col: [] for col in partition_by}
        result[time_column] = []
        result[feature_name] = []

        for key, indices in groups.items():
            first_idx = indices[0]
            for col_name, col in zip(partition_by, partition_cols):
                result[col_name].append(col[first_idx])
            result[time_column].append(key[-1])
            values = [source_values[i] for i in indices]
            result[feature_name].append(_reduce(agg, values))

        return result
