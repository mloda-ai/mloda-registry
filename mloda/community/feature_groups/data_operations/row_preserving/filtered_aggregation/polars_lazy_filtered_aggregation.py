"""Polars Lazy implementation for filtered aggregation feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

import polars as pl

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.filtered_aggregation.base import (
    FilteredAggregationFeatureGroup,
)

_POLARS_AGG_EXPRS: dict[str, Any] = {
    "sum": lambda col: col.sum(),
    "avg": lambda col: col.mean(),
    "mean": lambda col: col.mean(),
    "count": lambda col: col.count(),
    "min": lambda col: col.min(),
    "max": lambda col: col.max(),
}


class PolarsLazyFilteredAggregation(FilteredAggregationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PolarsLazyDataFrame}

    @classmethod
    def _compute_filtered(
        cls,
        data: pl.LazyFrame,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
        filter_column: str,
        filter_value: Any,
    ) -> pl.LazyFrame:
        """Compute a filtered aggregation using Polars expressions (fully lazy)."""
        agg_builder = _POLARS_AGG_EXPRS.get(agg_type)
        if agg_builder is None:
            raise ValueError(f"Unsupported aggregation type: {agg_type}")

        # Mask: source value where filter matches, null otherwise.
        filter_match = pl.col(filter_column) == filter_value
        masked = pl.when(filter_match).then(pl.col(source_col)).otherwise(None)

        if agg_type in ("sum",):
            # Polars returns 0 for sum of all-null groups. Replace with null
            # when no non-null values exist in the partition (PyArrow convergence).
            count_expr = masked.count().over(partition_by)
            sum_expr = masked.sum().over(partition_by)
            expr = pl.when(count_expr > 0).then(sum_expr).otherwise(None).alias(feature_name)
        else:
            expr = agg_builder(masked).over(partition_by).alias(feature_name)

        return data.with_columns(expr)
