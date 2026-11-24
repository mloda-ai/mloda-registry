"""Polars Lazy implementation for single-column global aggregate broadcast."""

from __future__ import annotations

from typing import Any

import polars as pl
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error
from mloda.community.feature_groups.data_operations.mask_utils import apply_polars_mask
from mloda.community.feature_groups.data_operations.polars_helpers import nan_to_null, two_pass_variance
from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.base import (
    ScalarAggregateFeatureGroup,
)


class PolarsLazyScalarAggregate(Implementation[ScalarAggregateFeatureGroup]):
    feature_group = ScalarAggregateFeatureGroup
    compute_framework = PolarsLazyDataFrame

    @classmethod
    def calculate(
        cls,
        group: type[ScalarAggregateFeatureGroup],
        data: Any,
        features: FeatureSet,
        framework: ComputeFramework | None,
    ) -> Any:
        return group.aggregate(data, features, compute_aggregation=cls._compute_aggregation)

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
            data, actual_source = apply_polars_mask(data, source_col, mask_spec, taken={feature_name})

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
            expr = col.count().cast(pl.Int64)
        elif agg_type in ("std", "std_pop", "std_samp", "var", "var_pop", "var_samp"):
            ddof = 1 if agg_type.endswith("_samp") else 0
            expr = two_pass_variance(col, ddof, agg_type.startswith("std"))
        elif agg_type == "median":
            dtype = data.collect_schema()[actual_source]
            expr = nan_to_null(col, dtype).median()
        else:
            raise unsupported_agg_type_error(
                agg_type, ScalarAggregateFeatureGroup._SUPPORTED_AGG_TYPES, framework="Polars"
            )

        result = data.with_columns(expr.alias(feature_name))
        if mask_spec is not None:
            result = result.drop(actual_source)
        return result
