"""PyArrow implementation of time bucketization (production AND reference)."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.base import (
    TIME_BUCKETIZATION_OPS,
    TimeBucketizationFeatureGroup,
)


class PyArrowTimeBucketization(TimeBucketizationFeatureGroup):
    """PyArrow backend; also the cross-framework reference implementation.

    Uses ``pyarrow.compute.{floor,ceil,round}_temporal`` with
    ``week_starts_monday=True`` (ISO) and ``ceil_is_strictly_greater=False``
    (idempotent on aligned). PyArrow's ``round_temporal`` default tie-break
    is half-up (every midpoint rounds toward the next bucket), which is the
    behaviour pinned for all backends.
    """

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PyArrowTable}

    @classmethod
    def _assert_source_column_is_timestamp(cls, data: pa.Table, source_col: str) -> None:
        if source_col not in data.schema.names:
            raise ValueError(
                f"Source column {source_col!r} is not present in the PyArrow table; "
                f"available: {data.schema.names}."
            )
        arrow_type = data.column(source_col).type
        if not pa.types.is_timestamp(arrow_type):
            cls._raise_non_timestamp_source(source_col, arrow_type)

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
