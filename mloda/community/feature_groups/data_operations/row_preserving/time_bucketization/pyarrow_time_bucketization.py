"""PyArrow implementation of time bucketization (production AND reference)."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.base import (
    TIME_BUCKETIZATION_OPS,
    TimeBucketizationFeatureGroup,
)


class PyArrowTimeBucketization(Implementation[TimeBucketizationFeatureGroup]):
    """PyArrow backend; also the cross-framework reference implementation.

    Uses ``pyarrow.compute.{floor,ceil,round}_temporal`` with
    ``week_starts_monday=True`` (ISO) and ``ceil_is_strictly_greater=False``
    (idempotent on aligned). PyArrow's ``round_temporal`` default tie-break
    is half-up (every midpoint rounds toward the next bucket), which is the
    behaviour pinned for all backends.
    """

    feature_group = TimeBucketizationFeatureGroup
    compute_framework = PyArrowTable

    @classmethod
    def calculate(
        cls,
        group: type[TimeBucketizationFeatureGroup],
        data: Any,
        features: FeatureSet,
        framework: ComputeFramework | None,
    ) -> Any:
        return group.bucketize(
            data, features, assert_timestamp=cls._assert_source_column_is_timestamp, compute_bucket=cls._compute_bucket
        )

    @classmethod
    def _assert_source_column_is_timestamp(cls, data: pa.Table, source_col: str) -> None:
        arrow_type = data.column(source_col).type
        if not pa.types.is_timestamp(arrow_type):
            TimeBucketizationFeatureGroup._raise_non_timestamp_source(source_col, arrow_type)

    @classmethod
    def _compute_bucket(
        cls,
        data: pa.Table,
        feature_name: str,
        source_col: str,
        op: str,
        n: int,
        unit: str,
    ) -> pa.Table:
        column = data.column(source_col)

        if op == "floor":
            result = pc.floor_temporal(column, multiple=n, unit=unit, week_starts_monday=True)
        elif op == "ceil":
            result = pc.ceil_temporal(
                column,
                multiple=n,
                unit=unit,
                week_starts_monday=True,
                ceil_is_strictly_greater=False,
            )
        elif op == "round":
            result = pc.round_temporal(column, multiple=n, unit=unit, week_starts_monday=True)
        else:
            raise ValueError(f"Unsupported bucket op {op!r} for PyArrow; supported: {sorted(TIME_BUCKETIZATION_OPS)}.")

        return data.append_column(feature_name, result)
