"""Polars Lazy implementation for single-column global aggregate broadcast."""

from __future__ import annotations

from typing import Any

import polars as pl

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.base import (
    ScalarAggregateFeatureGroup,
)


_MASK_TMP = "__mloda_masked_src__"


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
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> pl.LazyFrame:
        actual_source = source_col
        if mask_spec is not None:
            from mloda.community.feature_groups.data_operations.mask_utils import build_polars_mask_expr

            mask_expr = build_polars_mask_expr(mask_spec)
            data = data.with_columns(pl.when(mask_expr).then(pl.col(source_col)).otherwise(None).alias(_MASK_TMP))
            actual_source = _MASK_TMP

        col = pl.col(actual_source)

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

        result = data.with_columns(expr.alias(feature_name))
        if mask_spec is not None:
            result = result.drop(_MASK_TMP)
        return result
