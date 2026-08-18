"""PythonDict implementation of resample.

Floors each row's ``time_column`` to its ``(n, unit)`` bucket (mirrors
``python_dict_time_bucketization._floor_dt`` for minute/hour/day; see that
module for the epoch-anchor rationale), groups row indices by
``(*partition_by, bucket_start)``, then reduces each group's ``source_col``
values with the requested aggregation, skipping nulls. A non-empty bucket
with no non-null values still emits: ``count = 0``, other aggs ``None``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error
from mloda.community.feature_groups.data_operations.row_changing.resample.base import (
    RESAMPLE_AGGS,
    ResampleFeatureGroup,
)

_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)

# Fixed-duration units bucketed by an epoch-anchored floor (see _floor_dt).
_SECONDS_PER_UNIT: dict[str, int] = {"minute": 60, "hour": 3600, "day": 86400}


def _wall_clock_epoch_seconds(dt: datetime) -> int:
    """Whole seconds between 1970-01-01T00:00:00 and *dt*'s wall-clock fields.

    Computed by discarding ``dt``'s real ``tzinfo`` and treating its
    calendar/clock fields as if they were already UTC. PyArrow's
    ``floor_temporal`` (the reference oracle for resample bucketing) buckets
    minute/hour/day by a timestamp's *local* wall-clock representation, not
    its true UTC instant: e.g. 08:37 local in a UTC+05:30 zone floors a
    5-hour bucket to 08:00 local, not to the real-instant-equivalent 08:30.
    Anchoring this wall-clock value to the Unix epoch (rather than resetting
    within the enclosing hour/day, the previous bug) makes the floor correct
    for any ``n``, including values that don't evenly divide 60 (minute) or
    24 (hour). See ``python_dict_time_bucketization`` for the identical
    helper this mirrors.
    """
    delta = dt.replace(tzinfo=timezone.utc) - _EPOCH_UTC
    return delta.days * 86400 + delta.seconds


def _floor_dt(dt: datetime, n: int, unit: str) -> datetime:
    """Floor ``dt`` to the start of its ``(n, unit)`` bucket, preserving tzinfo."""
    if unit in _SECONDS_PER_UNIT:
        bucket_seconds = n * _SECONDS_PER_UNIT[unit]
        floored_seconds = (_wall_clock_epoch_seconds(dt) // bucket_seconds) * bucket_seconds
        floored = _EPOCH_UTC + timedelta(seconds=floored_seconds)
        return floored.replace(tzinfo=dt.tzinfo)
    raise ValueError(f"Unsupported resample unit for PythonDict: {unit!r}")


def _reduce(agg: str, values: list[Any]) -> Any:
    """Reduce one bucket's raw (possibly null-containing) values to a single result."""
    non_null = [v for v in values if v is not None]

    if agg == "count":
        return len(non_null)
    if not non_null:
        # All-null but non-empty bucket: mean/sum/min/max -> None (PyArrow oracle).
        return None
    if agg == "sum":
        return sum(non_null)
    if agg == "mean":
        return sum(non_null) / len(non_null)
    if agg == "min":
        return min(non_null)
    if agg == "max":
        return max(non_null)

    raise unsupported_agg_type_error(agg, RESAMPLE_AGGS, framework="PythonDict")


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

        buckets = [_floor_dt(value, n, unit) for value in time_col]

        # Group row indices by (*partition_by, bucket_start), first-occurrence order.
        groups: dict[tuple[Any, ...], list[int]] = {}
        for i in range(len(time_col)):
            key = tuple(col[i] for col in partition_cols) + (buckets[i],)
            groups.setdefault(key, []).append(i)

        result: dict[str, list[Any]] = {col: [] for col in partition_by}
        result[time_column] = []
        result[feature_name] = []

        for key, indices in groups.items():
            for col_name, key_value in zip(partition_by, key[:-1]):
                result[col_name].append(key_value)
            result[time_column].append(key[-1])
            values = [source_values[i] for i in indices]
            result[feature_name].append(_reduce(agg, values))

        return result
