"""Polars lazy implementation of gap-threshold sessionization."""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.sessionization.base import (
    SessionizationFeatureGroup,
)

_RN_COL = "__mloda_rn__"


class PolarsLazySessionization(SessionizationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PolarsLazyDataFrame}

    @classmethod
    def _assert_source_column_present(cls, data: pl.LazyFrame, order_col: str) -> None:
        names = data.collect_schema().names()
        if order_col not in names:
            raise ValueError(f"Source column {order_col!r} is not present in the polars frame; available: {names}.")

    @classmethod
    def _compute_session(
        cls,
        data: pl.LazyFrame,
        feature_name: str,
        order_col: str,
        threshold_seconds: int,
        partition_by: list[str],
    ) -> pl.LazyFrame:
        # Tag original row order so we can restore it after sorting.
        data = data.with_row_index(_RN_COL)

        # Sort within partitions by order_col (nulls last) so each partition is a
        # contiguous, time-ordered slice.
        ordered = data.sort([*partition_by, order_col], nulls_last=True)

        gap = pl.col(order_col).diff()
        if partition_by:
            gap = gap.over(partition_by)

        threshold = timedelta(seconds=threshold_seconds)
        is_new = gap.is_null() | (gap > threshold)
        session = is_new.cast(pl.Int64).cum_sum() - 1
        ordered = ordered.with_columns(session.alias(feature_name))

        # Restore original row order and drop the helper column.
        return ordered.sort(_RN_COL).drop(_RN_COL)
