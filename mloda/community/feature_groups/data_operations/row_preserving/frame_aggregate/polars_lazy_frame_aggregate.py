"""Polars lazy implementation for frame aggregate feature groups."""

from __future__ import annotations

from typing import Any

import polars as pl

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.mask_utils import build_polars_mask_expr
from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.base import (
    FrameAggregateFeatureGroup,
)

_RN_COL = "__mloda_rn__"


class PolarsLazyFrameAggregate(FrameAggregateFeatureGroup):
    SUPPORTED_FRAME_TYPES = {"rolling", "cumulative", "expanding"}

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PolarsLazyDataFrame}

    @classmethod
    def _compute_frame(
        cls,
        data: pl.LazyFrame,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        order_by: str,
        agg_type: str,
        frame_type: str,
        frame_size: int | None = None,
        frame_unit: str | None = None,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> pl.LazyFrame:
        _mask_tmp = "__mloda_masked_src__"
        actual_source = source_col
        if mask_spec is not None:
            mask_expr = build_polars_mask_expr(mask_spec)
            data = data.with_columns(pl.when(mask_expr).then(pl.col(source_col)).otherwise(None).alias(_mask_tmp))
            actual_source = _mask_tmp

        # Tag rows with original position
        data = data.with_row_index(_RN_COL)

        # Sort within partitions by order_by (nulls last)
        sort_expr = pl.col(order_by).is_null().cast(pl.Int8)
        sorted_data = data.sort(sort_expr, order_by)

        col = pl.col(actual_source)

        if frame_type in ("cumulative", "expanding"):
            # forward_fill() after cumulative ops ensures null source values carry
            # forward the last valid aggregate instead of propagating null.
            if agg_type == "sum":
                expr = col.cum_sum().forward_fill().over(partition_by).alias(feature_name)
            elif agg_type == "min":
                expr = col.cum_min().forward_fill().over(partition_by).alias(feature_name)
            elif agg_type == "max":
                expr = col.cum_max().forward_fill().over(partition_by).alias(feature_name)
            elif agg_type == "count":
                expr = col.cum_count().over(partition_by).alias(feature_name)
            elif agg_type == "avg":
                cum_sum = col.cum_sum().forward_fill().over(partition_by)
                cum_count = col.cum_count().over(partition_by).cast(pl.Float64)
                expr = (cum_sum / cum_count).alias(feature_name)
            else:
                raise ValueError(f"Unsupported cumulative/expanding agg for Polars: {agg_type}")
        elif frame_type == "rolling":
            window = int(frame_size) if frame_size is not None else 1
            if agg_type == "sum":
                expr = col.rolling_sum(window_size=window, min_samples=1).over(partition_by).alias(feature_name)
            elif agg_type == "avg":
                expr = col.rolling_mean(window_size=window, min_samples=1).over(partition_by).alias(feature_name)
            elif agg_type == "min":
                expr = col.rolling_min(window_size=window, min_samples=1).over(partition_by).alias(feature_name)
            elif agg_type == "max":
                expr = col.rolling_max(window_size=window, min_samples=1).over(partition_by).alias(feature_name)
            elif agg_type == "std":
                expr = col.rolling_std(window_size=window, min_samples=2, ddof=0).over(partition_by).alias(feature_name)
            elif agg_type == "var":
                expr = col.rolling_var(window_size=window, min_samples=2, ddof=0).over(partition_by).alias(feature_name)
            elif agg_type == "median":
                expr = col.rolling_median(window_size=window, min_samples=1).over(partition_by).alias(feature_name)
            elif agg_type == "count":
                expr = (
                    col.is_not_null()
                    .cast(pl.Int64)
                    .rolling_sum(window_size=window, min_samples=1)
                    .over(partition_by)
                    .alias(feature_name)
                )
            else:
                raise ValueError(f"Unsupported rolling agg for Polars: {agg_type}")
        else:
            raise ValueError(f"Unsupported frame type for Polars: {frame_type}")

        result = sorted_data.with_columns(expr)

        # Restore original row order and drop helper columns
        result = result.sort(_RN_COL)
        drop_cols = [_RN_COL]
        if mask_spec is not None:
            drop_cols.append(_mask_tmp)
        result = result.drop(drop_cols)

        return result
