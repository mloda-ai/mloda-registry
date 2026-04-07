"""Polars Lazy implementation for window aggregation feature groups."""

from __future__ import annotations

from typing import Any

import polars as pl

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.mask_utils import build_polars_mask_expr
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
    WindowAggregationFeatureGroup,
)

_MASK_TMP = "__mloda_masked_src__"

# Mapping from aggregation type to a Polars expression builder.
# Each callable takes a column name and returns a Polars Expr.
_POLARS_AGG_EXPRS: dict[str, Any] = {
    "sum": lambda col: pl.col(col).sum(),
    "avg": lambda col: pl.col(col).mean(),
    "mean": lambda col: pl.col(col).mean(),
    "count": lambda col: pl.col(col).count(),
    "min": lambda col: pl.col(col).min(),
    "max": lambda col: pl.col(col).max(),
    "std": lambda col: pl.col(col).std(ddof=0),
    "var": lambda col: pl.col(col).var(ddof=0),
    "std_pop": lambda col: pl.col(col).std(ddof=0),
    "std_samp": lambda col: pl.col(col).std(ddof=1),
    "var_pop": lambda col: pl.col(col).var(ddof=0),
    "var_samp": lambda col: pl.col(col).var(ddof=1),
    "median": lambda col: pl.col(col).median(),
    "nunique": lambda col: pl.col(col).drop_nulls().n_unique(),
}


class PolarsLazyWindowAggregation(WindowAggregationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PolarsLazyDataFrame}

    @classmethod
    def _compute_window(
        cls,
        data: pl.LazyFrame,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
        order_by: str | None = None,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> pl.LazyFrame:
        """Compute a window aggregation using Polars .over() expressions (fully lazy)."""
        actual_source = source_col
        if mask_spec is not None:
            mask_expr = build_polars_mask_expr(mask_spec)
            data = data.with_columns(pl.when(mask_expr).then(pl.col(source_col)).otherwise(None).alias(_MASK_TMP))
            actual_source = _MASK_TMP

        if agg_type == "mode":
            expr = pl.col(actual_source).mode().first().over(partition_by).alias(feature_name)
        elif agg_type in ("first", "last"):
            expr = cls._build_first_last_expr(actual_source, partition_by, agg_type, order_by, feature_name)
        elif agg_type in _POLARS_AGG_EXPRS:
            raw_expr = _POLARS_AGG_EXPRS[agg_type](actual_source).over(partition_by)
            if mask_spec is not None and agg_type == "sum":
                # Polars sum() returns 0 for all-null groups; correct to null.
                has_values = pl.col(actual_source).count().over(partition_by) > 0
                expr = pl.when(has_values).then(raw_expr).otherwise(None).alias(feature_name)
            else:
                expr = raw_expr.alias(feature_name)
        else:
            raise ValueError(f"Unsupported aggregation type: {agg_type}")

        result = data.with_columns(expr)
        if mask_spec is not None:
            result = result.drop(_MASK_TMP)
        return result

    @classmethod
    def _build_first_last_expr(
        cls,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
        order_by: str | None,
        feature_name: str,
    ) -> pl.Expr:
        """Build a Polars expression for first/last with deterministic ordering."""
        base = pl.col(source_col)
        if order_by:
            base = base.sort_by(order_by)
        base = base.drop_nulls()
        if agg_type == "first":
            return base.first().over(partition_by).alias(feature_name)
        return base.last().over(partition_by).alias(feature_name)
