"""Polars Lazy implementation for aggregation feature groups."""

from __future__ import annotations

from typing import Any

import polars as pl

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.aggregation.base import (
    AggregationFeatureGroup,
)
from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error
from mloda.community.feature_groups.data_operations.mask_utils import _POLARS_MASK_TMP, apply_polars_mask
from mloda.community.feature_groups.data_operations.polars_agg_constants import POLARS_AGG_EXPRS
from mloda.community.feature_groups.data_operations.polars_mode_helpers import (
    ModeHelperCols,
    add_mode_helper_cols,
    mode_agg_expr,
)

# Aggregation extends the shared builders with first/last.
_POLARS_AGG_EXPRS: dict[str, Any] = {
    **POLARS_AGG_EXPRS,
    "first": lambda col: pl.col(col).drop_nulls().first(),
    "last": lambda col: pl.col(col).drop_nulls().last(),
}

_SUPPORTED_AGG_TYPES = {*_POLARS_AGG_EXPRS.keys(), "mode"}


class PolarsLazyAggregation(AggregationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PolarsLazyDataFrame}

    @classmethod
    def _compute_group(
        cls,
        data: pl.LazyFrame,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> pl.LazyFrame:
        actual_source = source_col
        if mask_spec is not None:
            data, actual_source = apply_polars_mask(data, source_col, mask_spec)

        if agg_type == "mode":
            cols = ModeHelperCols.pick(set(data.collect_schema().names()) | {feature_name})
            data = add_mode_helper_cols(data, actual_source, partition_by, cols)
            expr = mode_agg_expr(actual_source, feature_name, cols)
        elif agg_type in _POLARS_AGG_EXPRS:
            raw_expr = _POLARS_AGG_EXPRS[agg_type](actual_source).alias(feature_name)
            if agg_type == "sum":
                # Polars sum() returns 0 for all-null groups; correct to null.
                has_values = pl.col(actual_source).count() > 0
                expr = (
                    pl.when(has_values)
                    .then(_POLARS_AGG_EXPRS[agg_type](actual_source))
                    .otherwise(None)
                    .alias(feature_name)
                )
            else:
                expr = raw_expr
        else:
            raise unsupported_agg_type_error(agg_type, _SUPPORTED_AGG_TYPES, framework="Polars")

        result = data.group_by(partition_by, maintain_order=True).agg(expr)
        if mask_spec is not None:
            result = result.drop(_POLARS_MASK_TMP, strict=False)
        return result
