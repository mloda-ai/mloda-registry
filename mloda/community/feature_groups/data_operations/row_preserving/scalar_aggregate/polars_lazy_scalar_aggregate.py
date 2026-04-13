"""Polars Lazy implementation for single-column global aggregate broadcast."""

from __future__ import annotations

from typing import Any

import polars as pl

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error
from mloda.community.feature_groups.data_operations.mask_utils import _POLARS_MASK_TMP, apply_polars_mask
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
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> pl.LazyFrame:
        actual_source = source_col
        if mask_spec is not None:
            data, actual_source = apply_polars_mask(data, source_col, mask_spec)

        col = pl.col(actual_source)

        if agg_type == "sum":
            raw = col.sum()
            # Polars sum() returns 0 for all-null columns; correct to null.
            has_values = pl.col(actual_source).count() > 0
            expr = pl.when(has_values).then(raw).otherwise(None)
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
            raise unsupported_agg_type_error(agg_type, cls._SUPPORTED_AGG_TYPES, framework="Polars")

        result = data.with_columns(expr.alias(feature_name))
        if mask_spec is not None:
            result = result.drop(_POLARS_MASK_TMP)
        return result
