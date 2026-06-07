"""Polars lazy implementation of EMA-by-time."""

from __future__ import annotations

import polars as pl

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.helper_columns import unique_helper_name
from mloda.community.feature_groups.data_operations.row_preserving.ema.base import EmaFeatureGroup


class PolarsLazyEma(EmaFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PolarsLazyDataFrame}

    @classmethod
    def _assert_source_column_present(cls, data: pl.LazyFrame, source_col: str) -> None:
        names = data.collect_schema().names()
        if source_col not in names:
            raise ValueError(f"Source column {source_col!r} is not present in the polars frame; available: {names}.")

    @classmethod
    def _compute_ema(
        cls,
        data: pl.LazyFrame,
        feature_name: str,
        source_col: str,
        span: int,
        partition_by: list[str],
        order_by: str,
    ) -> pl.LazyFrame:
        rn_col = unique_helper_name("__mloda_rn__", set(data.collect_schema().names()) | {feature_name})

        # Tag original row order so we can restore it after sorting.
        data = data.with_row_index(rn_col)

        # Sort within partitions by order_by (nulls last).
        sort_null_last = pl.col(order_by).is_null().cast(pl.Int8)
        ordered = data.sort(sort_null_last, order_by)

        # adjust=False, nulls skipped in the recurrence; null input -> null output.
        ema_expr = pl.col(source_col).ewm_mean(span=span, adjust=False, ignore_nulls=True)
        if partition_by:
            ema_expr = ema_expr.over(partition_by)
        ordered = ordered.with_columns(ema_expr.alias(feature_name))

        # Restore original row order and drop the helper column.
        return ordered.sort(rn_col).drop(rn_col)
