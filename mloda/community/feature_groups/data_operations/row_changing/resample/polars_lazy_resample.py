"""Polars Lazy implementation of resample.

Floors the time column with ``dt.truncate`` (same duration tokens as
``polars_lazy_time_bucketization``), groups by ``(*partition_by, bucket)`` and
aggregates. Polars ``sum()`` returns ``0`` for an all-null group, so the sum
path coerces all-null buckets to ``None`` to match the PyArrow oracle.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.row_changing.resample.base import (
    RESAMPLE_AGGS,
    ResampleFeatureGroup,
)
from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.polars_lazy_time_bucketization import (
    _duration_token,
)

# Resample agg -> Polars expression builder over the source column.
_POLARS_AGG_EXPRS: dict[str, Any] = {
    "mean": lambda col: pl.col(col).mean(),
    "sum": lambda col: pl.col(col).sum(),
    "count": lambda col: pl.col(col).count(),
    "min": lambda col: pl.col(col).min(),
    "max": lambda col: pl.col(col).max(),
}


class PolarsLazyResample(ResampleFeatureGroup):
    """Polars-lazy backend for resample."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PolarsLazyDataFrame}

    @classmethod
    def _assert_time_column_present(cls, data: pl.LazyFrame, time_column: str) -> None:
        schema = data.collect_schema()
        if time_column not in schema:
            raise ValueError(
                f"time_column {time_column!r} is not present in the Polars LazyFrame; available: {list(schema)}."
            )

    @classmethod
    def _assert_source_column_present(cls, data: pl.LazyFrame, source_col: str) -> None:
        schema = data.collect_schema()
        if source_col not in schema:
            raise ValueError(
                f"Source column {source_col!r} is not present in the Polars LazyFrame; available: {list(schema)}."
            )

    @classmethod
    def _compute_resample(
        cls,
        data: pl.LazyFrame,
        feature_name: str,
        source_col: str,
        time_column: str,
        partition_by: list[str],
        n: int,
        unit: str,
        agg: str,
    ) -> pl.LazyFrame:
        if agg not in _POLARS_AGG_EXPRS:
            raise ValueError(f"Unsupported resample agg {agg!r} for Polars; supported: {sorted(RESAMPLE_AGGS)}.")

        duration = _duration_token(n, unit)
        # Floor the time column in place (bucket start keeps the original name).
        data = data.with_columns(pl.col(time_column).dt.truncate(duration).alias(time_column))

        if agg == "sum":
            # Polars sum() returns 0 for all-null groups; correct to None.
            has_values = pl.col(source_col).count() > 0
            expr = pl.when(has_values).then(pl.col(source_col).sum()).otherwise(None).alias(feature_name)
        else:
            expr = _POLARS_AGG_EXPRS[agg](source_col).alias(feature_name)

        keys = [*partition_by, time_column]
        # maintain_order=True so repeated ``.collect()`` calls (the test extracts
        # each column separately) return a stable row order, keeping bucket keys
        # aligned with their aggregate values.
        return data.group_by(keys, maintain_order=True).agg(expr)
