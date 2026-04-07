"""Polars Lazy implementation for rank feature groups."""

from __future__ import annotations


import polars as pl

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.rank.base import (
    RankFeatureGroup,
)

_NULL_FLAG_COL = "__mloda_rank_null_flag__"


class PolarsLazyRank(RankFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PolarsLazyDataFrame}

    @classmethod
    def _row_number_nulls_last(
        cls,
        order_by: str,
        partition_by: list[str],
        descending: bool = False,
    ) -> pl.Expr:
        """Row number expression with nulls-last semantics.

        Returns an Int64 expression giving each row its 1-based position
        within its partition, with null values in *order_by* ranked after
        all non-null values.
        """
        non_null_rank = pl.col(order_by).rank(method="ordinal", descending=descending).over(partition_by)
        group_size = pl.col(order_by).len().over(partition_by)
        null_count_before = pl.col(_NULL_FLAG_COL).cum_sum().over(partition_by)
        non_null_count = group_size - pl.col(_NULL_FLAG_COL).sum().over(partition_by)
        return pl.when(pl.col(order_by).is_null()).then(non_null_count + null_count_before).otherwise(non_null_rank)

    @classmethod
    def _compute_rank(
        cls,
        data: pl.LazyFrame,
        feature_name: str,
        partition_by: list[str],
        order_by: str,
        rank_type: str,
    ) -> pl.LazyFrame:
        """Compute rank using Polars expressions (fully lazy).

        NullPolicy.NULLS_LAST: nulls in order_by get the highest rank.
        Polars rank() returns null for null inputs, so we handle nulls
        by assigning them a rank of (group_size) or (group_size + 1).
        """
        # Create a helper: is_null flag (0 for non-null, 1 for null) for sorting nulls last
        null_flag = pl.col(order_by).is_null().cast(pl.Int64).alias(_NULL_FLAG_COL)
        data = data.with_columns(null_flag)

        if rank_type == "row_number":
            row_num = cls._row_number_nulls_last(order_by, partition_by)
            expr = row_num.cast(pl.Int64).alias(feature_name)
        elif rank_type == "rank":
            non_null_rank = pl.col(order_by).rank(method="min").over(partition_by)
            group_size = pl.col(order_by).len().over(partition_by)
            non_null_count = group_size - pl.col(_NULL_FLAG_COL).sum().over(partition_by)
            expr = (
                pl.when(pl.col(order_by).is_null())
                .then(non_null_count + 1)
                .otherwise(non_null_rank)
                .cast(pl.Int64)
                .alias(feature_name)
            )
        elif rank_type == "dense_rank":
            non_null_rank = pl.col(order_by).rank(method="dense").over(partition_by)
            n_unique = pl.col(order_by).drop_nulls().n_unique().over(partition_by)
            expr = (
                pl.when(pl.col(order_by).is_null())
                .then(n_unique + 1)
                .otherwise(non_null_rank)
                .cast(pl.Int64)
                .alias(feature_name)
            )
        elif rank_type == "percent_rank":
            non_null_rank = pl.col(order_by).rank(method="min").over(partition_by)
            group_size = pl.col(order_by).len().over(partition_by)
            non_null_count = group_size - pl.col(_NULL_FLAG_COL).sum().over(partition_by)
            null_rank = non_null_count + 1
            rank_val = pl.when(pl.col(order_by).is_null()).then(null_rank).otherwise(non_null_rank)
            expr = (
                pl.when(group_size == 1)
                .then(pl.lit(0.0))
                .otherwise((rank_val.cast(pl.Float64) - 1.0) / (group_size.cast(pl.Float64) - 1.0))
                .alias(feature_name)
            )
        elif rank_type.startswith("ntile_"):
            ntile_n = int(rank_type[len("ntile_") :])
            row_num = cls._row_number_nulls_last(order_by, partition_by)
            group_size = pl.col(order_by).len().over(partition_by)
            expr = ((row_num - 1) * ntile_n // group_size + 1).cast(pl.Int64).alias(feature_name)
        elif rank_type.startswith("top_"):
            top_n = int(rank_type[len("top_") :])
            row_num = cls._row_number_nulls_last(order_by, partition_by, descending=True)
            expr = (row_num <= top_n).alias(feature_name)
        elif rank_type.startswith("bottom_"):
            bottom_n = int(rank_type[len("bottom_") :])
            row_num = cls._row_number_nulls_last(order_by, partition_by)
            expr = (row_num <= bottom_n).alias(feature_name)
        else:
            raise ValueError(f"Unsupported rank type: {rank_type}")

        result = data.with_columns(expr)
        return result.drop(_NULL_FLAG_COL)
