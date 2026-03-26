"""Polars Lazy implementation for column aggregation feature groups."""

from __future__ import annotations

from typing import Set, Type, Union

import polars as pl

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.aggregation.base import (
    ColumnAggregationFeatureGroup,
)


class PolarsLazyColumnAggregation(ColumnAggregationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
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
        elif agg_type == "std":
            expr = col.std(ddof=0)
        elif agg_type == "var":
            expr = col.var(ddof=0)
        elif agg_type == "median":
            expr = col.median()
        else:
            raise ValueError(f"Unsupported aggregation type: {agg_type}")

        return data.with_columns(expr.alias(feature_name))
