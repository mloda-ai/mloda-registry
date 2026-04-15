"""Polars Lazy implementation for percentile feature groups."""

from __future__ import annotations

from typing import Any

import polars as pl

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.mask_utils import _POLARS_MASK_TMP, apply_polars_mask
from mloda.community.feature_groups.data_operations.reserved_columns import assert_no_reserved_columns
from mloda.community.feature_groups.data_operations.row_preserving.percentile.base import (
    PercentileFeatureGroup,
)


class PolarsLazyPercentile(PercentileFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PolarsLazyDataFrame}

    @classmethod
    def _compute_percentile(
        cls,
        data: pl.LazyFrame,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        percentile: float,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> pl.LazyFrame:
        assert_no_reserved_columns(data.collect_schema().names(), framework="Polars", operation="percentile")

        actual_source = source_col
        if mask_spec is not None:
            data, actual_source = apply_polars_mask(data, source_col, mask_spec)

        expr = pl.col(actual_source).quantile(percentile, interpolation="linear").over(partition_by).alias(feature_name)
        result = data.with_columns(expr)
        if mask_spec is not None:
            result = result.drop(_POLARS_MASK_TMP)
        return result
