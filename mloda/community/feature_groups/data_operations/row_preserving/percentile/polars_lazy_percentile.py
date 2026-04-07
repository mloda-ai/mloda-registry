"""Polars Lazy implementation for percentile feature groups."""

from __future__ import annotations

import polars as pl

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.percentile.base import (
    PercentileFeatureGroup,
)


class PolarsLazyPercentile(PercentileFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PolarsLazyDataFrame}

    @classmethod
    def _compute_percentile(
        cls,
        data: pl.LazyFrame,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        percentile: float,
    ) -> pl.LazyFrame:
        expr = pl.col(source_col).quantile(percentile, interpolation="linear").over(partition_by).alias(feature_name)
        return data.with_columns(expr)
