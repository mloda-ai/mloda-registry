"""PyArrow implementation of time bucketization."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.pyarrow_helpers import assert_temporal_timezone_supported
from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.base import (
    TIME_BUCKETIZATION_OPS,
    TimeBucketizationFeatureGroup,
)


class PyArrowTimeBucketization(Implementation[TimeBucketizationFeatureGroup]):
    """PyArrow backend.

    Uses ``pyarrow.compute.{floor,ceil,round}_temporal`` with
    ``week_starts_monday=True`` (ISO). Ceil is idempotent on aligned minute/hour/day;
    week advances an aligned value. Round ties go up, as pinned for all backends.
    ``round_temporal`` serves every unit except month and year, which are composed from
    floor and the next boundary because it moves their midpoint by the UTC offset.
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
        assert_temporal_timezone_supported(arrow_type, source_col)

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
        if column.type.unit == "ns":
            cls._assert_boundaries_in_range(column, source_col, op, n, unit)

        if op == "floor":
            result = cls._floor(column, n, unit)
        elif op == "ceil":
            result = cls._ceil(column, n, unit)
        elif op == "round" and unit in ("month", "year"):
            floor = cls._floor(column, n, unit)
            upper = cls._ceil(column, n, unit)
            result = pc.if_else(pc.greater_equal(pc.subtract(column, floor), pc.subtract(upper, column)), upper, floor)
        elif op == "round":
            result = pc.round_temporal(column, multiple=n, unit=unit, week_starts_monday=True)
        else:
            raise ValueError(f"Unsupported bucket op {op!r} for PyArrow; supported: {sorted(TIME_BUCKETIZATION_OPS)}.")

        return data.append_column(feature_name, result)

    @classmethod
    def _assert_boundaries_in_range(cls, column: Any, source_col: str, op: str, n: int, unit: str) -> None:
        """Reject ns sources whose boundary wrapped modulo 2^64 (floor above, ceil / next boundary below the source)."""
        bad = pc.greater(cls._floor(column, n, unit), column)
        if op == "ceil":
            bad = pc.or_kleene(bad, pc.less(cls._ceil(column, n, unit), column))
        elif op == "round":
            upper = pc.ceil_temporal(
                column, multiple=n, unit=unit, week_starts_monday=True, ceil_is_strictly_greater=True
            )
            bad = pc.or_kleene(bad, pc.less(upper, column))
        bad = pc.fill_null(bad, False)
        if pc.any(bad).as_py():
            TimeBucketizationFeatureGroup._raise_boundary_out_of_range(source_col, pc.filter(column, bad)[0].as_py())

    @staticmethod
    def _floor(column: Any, n: int, unit: str) -> Any:
        return pc.floor_temporal(column, multiple=n, unit=unit, week_starts_monday=True)

    @staticmethod
    def _ceil(column: Any, n: int, unit: str) -> Any:
        return pc.ceil_temporal(
            column,
            multiple=n,
            unit=unit,
            week_starts_monday=True,
            ceil_is_strictly_greater=unit == "week",
        )
