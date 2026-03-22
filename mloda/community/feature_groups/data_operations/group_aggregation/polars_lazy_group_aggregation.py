"""Polars Lazy implementation for group aggregation feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

import polars as pl

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.group_aggregation.base import (
    GroupAggregationFeatureGroup,
)

# Mapping from aggregation type to a Polars expression builder.
_POLARS_AGG_EXPRS: dict[str, Any] = {
    "sum": lambda col: pl.col(col).sum(),
    "avg": lambda col: pl.col(col).mean(),
    "count": lambda col: pl.col(col).count(),
    "min": lambda col: pl.col(col).min(),
    "max": lambda col: pl.col(col).max(),
    "std": lambda col: pl.col(col).std(),
    "var": lambda col: pl.col(col).var(),
    "median": lambda col: pl.col(col).median(),
    "nunique": lambda col: pl.col(col).n_unique(),
    "first": lambda col: pl.col(col).first(),
    "last": lambda col: pl.col(col).last(),
}


class PolarsLazyGroupAggregation(GroupAggregationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PolarsLazyDataFrame}

    @classmethod
    def _compute_group(
        cls,
        data: pl.LazyFrame,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
    ) -> pl.LazyFrame:
        """Compute a group aggregation using Polars group_by().agg() (fully lazy)."""
        if agg_type == "mode":
            expr = pl.col(source_col).mode().first().alias(feature_name)
        elif agg_type in _POLARS_AGG_EXPRS:
            expr = _POLARS_AGG_EXPRS[agg_type](source_col).alias(feature_name)
        else:
            raise ValueError(f"Unsupported aggregation type: {agg_type}")

        return data.group_by(partition_by, maintain_order=True).agg(expr)
