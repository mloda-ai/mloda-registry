"""Polars Lazy implementation for single-column global aggregate broadcast."""

from __future__ import annotations

import polars as pl

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.base import (
    ScalarAggregateFeatureGroup,
)


class PolarsLazyScalarAggregate(ScalarAggregateFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PolarsLazyDataFrame}

    @classmethod
    def _compute_aggregation(
        cls,
        data: pl.LazyFrame,
        feature_name: str,
        source_col: str,
        agg_type: str,
    ) -> pl.LazyFrame:
        col = pl.col(source_col)

        if agg_type == "sum":
            expr = col.sum()
        elif agg_type == "min":
            expr = col.min()
        elif agg_type == "max":
            expr = col.max()
        elif agg_type in ("avg", "mean"):
            expr = col.mean()
        elif agg_type == "count":
            expr = col.count()
        elif agg_type in ("std", "std_pop"):
            expr = col.std(ddof=0)
        elif agg_type in ("var", "var_pop"):
            expr = col.var(ddof=0)
        elif agg_type == "std_samp":
            expr = col.std(ddof=1)
        elif agg_type == "var_samp":
            expr = col.var(ddof=1)
        elif agg_type == "median":
            expr = col.median()
        else:
            raise ValueError(f"Unsupported aggregation type: {agg_type}")

        return data.with_columns(expr.alias(feature_name))
