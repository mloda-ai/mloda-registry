"""Polars lazy implementation of gap-threshold sessionization."""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.helper_columns import unique_helper_name
from mloda.community.feature_groups.data_operations.polars_helpers import assert_source_col_present
from mloda.community.feature_groups.data_operations.row_preserving.sessionization.base import (
    SessionizationFeatureGroup,
)


class PolarsLazySessionization(SessionizationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PolarsLazyDataFrame}

    @classmethod
    def _assert_source_column_present(cls, data: pl.LazyFrame, order_col: str) -> None:
        assert_source_col_present(data, order_col)

    @classmethod
    def _compute_session(
        cls,
        data: pl.LazyFrame,
        feature_name: str,
        order_col: str,
        threshold_seconds: int,
        partition_by: list[str],
    ) -> pl.LazyFrame:
        rn_col = unique_helper_name("__mloda_rn__", set(data.collect_schema().names()) | {feature_name})

        # Tag original row order so we can restore it after sorting.
        data = data.with_row_index(rn_col)

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
        return ordered.sort(rn_col).drop(rn_col)
