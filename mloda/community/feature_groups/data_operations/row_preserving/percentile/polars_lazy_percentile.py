"""Polars Lazy implementation for percentile feature groups."""

from __future__ import annotations

from typing import Any

import polars as pl

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.mask_utils import build_polars_mask_expr
from mloda.community.feature_groups.data_operations.row_preserving.percentile.base import (
    PercentileFeatureGroup,
)

_MASK_TMP = "__mloda_masked_pctl__"


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
        actual_source = source_col
        if mask_spec is not None:
            mask_expr = build_polars_mask_expr(mask_spec)
            data = data.with_columns(pl.when(mask_expr).then(pl.col(source_col)).otherwise(None).alias(_MASK_TMP))
            actual_source = _MASK_TMP

        expr = pl.col(actual_source).quantile(percentile, interpolation="linear").over(partition_by).alias(feature_name)
        result = data.with_columns(expr)
        if mask_spec is not None:
            result = result.drop(_MASK_TMP)
        return result
