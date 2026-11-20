"""Polars Lazy implementation of time bucketization.

Floor is the native ``dt.truncate``. Polars has no ``ceil_temporal`` and its
``dt.round`` does not match the reference midpoint rule on calendar units, so
ceil and round are built from the floor and ``floor.offset_by(duration)``:
ceil keeps an aligned fixed-freq value and always advances a calendar one;
round takes the upper boundary when ``2 * (col - floor) >= upper - floor``.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.polars_helpers import duration_token
from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.base import (
    TIME_BUCKETIZATION_OPS,
    TimeBucketizationFeatureGroup,
)

# Calendar units whose ``ceil`` always advances by one bucket even on
# aligned input (matches PyArrow's ``ceil_temporal`` behaviour for
# ``week`` / ``month`` / ``year``). Fixed-freq units are idempotent on
# aligned input.
_CALENDAR_CEIL_ALWAYS_ADVANCES: frozenset[str] = frozenset({"week", "month", "year"})


class PolarsLazyTimeBucketization(Implementation[TimeBucketizationFeatureGroup]):
    """Polars-lazy backend for time bucketization."""

    feature_group = TimeBucketizationFeatureGroup
    compute_framework = PolarsLazyDataFrame

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
    def _assert_source_column_is_timestamp(cls, data: pl.LazyFrame, source_col: str) -> None:
        schema = data.collect_schema()
        dtype = schema[source_col]
        if not (dtype == pl.Datetime or (hasattr(dtype, "base_type") and dtype.base_type() == pl.Datetime)):
            TimeBucketizationFeatureGroup._raise_non_timestamp_source(source_col, dtype)

    @classmethod
    def _compute_bucket(
        cls,
        data: pl.LazyFrame,
        feature_name: str,
        source_col: str,
        op: str,
        n: int,
        unit: str,
    ) -> pl.LazyFrame:
        col = pl.col(source_col)
        duration = duration_token(n, unit)

        if op == "floor":
            expr = col.dt.truncate(duration)
        elif op == "ceil":
            expr = cls._ceil_expr(col, duration, unit)
        elif op == "round":
            floored, upper = cls._floor_and_upper(col, duration)
            expr = pl.when(2 * (col - floored) >= upper - floored).then(upper).otherwise(floored)
        else:
            raise ValueError(f"Unsupported bucket op {op!r} for Polars; supported: {sorted(TIME_BUCKETIZATION_OPS)}.")

        return data.with_columns(expr.alias(feature_name))

    @classmethod
    def _ceil_expr(cls, col: pl.Expr, duration: str, unit: str) -> pl.Expr:
        """Polars ceil expression matching PyArrow's quirky calendar-unit behaviour."""
        floored, upper = cls._floor_and_upper(col, duration)
        if unit in _CALENDAR_CEIL_ALWAYS_ADVANCES:
            # Calendar units always advance even on aligned input.
            return upper
        # Fixed-freq units: idempotent on aligned input.
        return pl.when(col == floored).then(floored).otherwise(upper)

    @staticmethod
    def _floor_and_upper(col: pl.Expr, duration: str) -> tuple[pl.Expr, pl.Expr]:
        """Floor of ``col`` and the next bucket boundary above that floor."""
        floored = col.dt.truncate(duration)
        return floored, floored.dt.offset_by(duration)
