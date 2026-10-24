"""Polars Lazy implementation for aggregation feature groups."""

from __future__ import annotations

from typing import Any

import polars as pl
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.aggregation.base import (
    AggregationFeatureGroup,
    AggregationSpec,
)
from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error
from mloda.community.feature_groups.data_operations.helper_columns import unique_helper_name
from mloda.community.feature_groups.data_operations.mask_utils import _POLARS_MASK_TMP, apply_polars_mask
from mloda.community.feature_groups.data_operations.polars_helpers import nan_to_null
from mloda.community.feature_groups.data_operations.polars_mode_helpers import (
    ModeHelperCols,
    add_mode_helper_cols,
    mode_agg_expr,
)

# Mapping from aggregation type to a Polars expression builder.
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
    "nunique": lambda col: pl.col(col).drop_nulls().n_unique(),
    "first": lambda col: pl.col(col).drop_nulls().first(),
    "last": lambda col: pl.col(col).drop_nulls().last(),
}

_SUPPORTED_AGG_TYPES = {*_POLARS_AGG_EXPRS.keys(), "mode", "median"}


class PolarsLazyAggregation(Implementation[AggregationFeatureGroup]):
    feature_group = AggregationFeatureGroup
    compute_framework = PolarsLazyDataFrame

    @classmethod
    def calculate(
        cls, group: type[AggregationFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.aggregate(data, features, compute_group=cls._compute_group)

    @classmethod
    def _compute_group(cls, data: pl.LazyFrame, specs: list[AggregationSpec], partition_by: list[str]) -> pl.LazyFrame:
        for spec in specs:
            if spec.agg_type not in _SUPPORTED_AGG_TYPES:
                raise unsupported_agg_type_error(spec.agg_type, _SUPPORTED_AGG_TYPES, framework="Polars")

        feature_names = {spec.feature_name for spec in specs}
        exprs: list[pl.Expr] = []
        for spec in specs:
            feature_name = spec.feature_name
            actual_source = spec.source_col
            if spec.mask_spec is not None:
                # Temp masked column per spec; its mask reads only original columns.
                tmp = unique_helper_name(_POLARS_MASK_TMP, set(data.collect_schema().names()) | feature_names)
                data, actual_source = apply_polars_mask(data, spec.source_col, spec.mask_spec, tmp)

            if spec.agg_type == "mode":
                cols = ModeHelperCols.pick(set(data.collect_schema().names()) | feature_names)
                data = add_mode_helper_cols(data, actual_source, partition_by, cols)
                exprs.append(mode_agg_expr(actual_source, feature_name, cols))
            elif spec.agg_type == "median":
                dtype = data.collect_schema()[actual_source]
                exprs.append(nan_to_null(pl.col(actual_source), dtype).median().alias(feature_name))
            elif spec.agg_type == "sum":
                # Polars sum() returns 0 for all-null groups; correct to null.
                has_values = pl.col(actual_source).count() > 0
                exprs.append(
                    pl.when(has_values)
                    .then(_POLARS_AGG_EXPRS["sum"](actual_source))
                    .otherwise(None)
                    .alias(feature_name)
                )
            else:
                exprs.append(_POLARS_AGG_EXPRS[spec.agg_type](actual_source).alias(feature_name))

        return data.group_by(partition_by, maintain_order=True).agg(*exprs)
