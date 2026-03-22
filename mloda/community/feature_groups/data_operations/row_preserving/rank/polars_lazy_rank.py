"""Polars Lazy implementation for rank feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

import polars as pl

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.rank.base import (
    RankFeatureGroup,
)


class PolarsLazyRank(RankFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PolarsLazyDataFrame}

    @classmethod
    def _compute_rank(
        cls,
        data: pl.LazyFrame,
        feature_name: str,
        source_col: str,
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
        null_flag = pl.col(order_by).is_null().cast(pl.Int64).alias("__mloda_null_flag")
        data = data.with_columns(null_flag)

        if rank_type == "row_number":
            # For non-null: rank by value. For null: assign rank = group_size.
            non_null_rank = pl.col(order_by).rank(method="ordinal").over(partition_by)
            group_size = pl.col(order_by).len().over(partition_by)
            # null_rank_offset: count of non-null values + position among nulls
            null_count_before = pl.col("__mloda_null_flag").cum_sum().over(partition_by)
            non_null_count = group_size - pl.col("__mloda_null_flag").sum().over(partition_by)

            expr = (
                pl.when(pl.col(order_by).is_null())
                .then(non_null_count + null_count_before)
                .otherwise(non_null_rank)
                .cast(pl.Int64)
                .alias(feature_name)
            )
        elif rank_type == "rank":
            non_null_rank = pl.col(order_by).rank(method="min").over(partition_by)
            group_size = pl.col(order_by).len().over(partition_by)
            non_null_count = group_size - pl.col("__mloda_null_flag").sum().over(partition_by)
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
            non_null_count = group_size - pl.col("__mloda_null_flag").sum().over(partition_by)
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
            # Use row_number approach for ntile
            non_null_rank = pl.col(order_by).rank(method="ordinal").over(partition_by)
            group_size = pl.col(order_by).len().over(partition_by)
            non_null_count = group_size - pl.col("__mloda_null_flag").sum().over(partition_by)
            null_count_before = pl.col("__mloda_null_flag").cum_sum().over(partition_by)
            row_num = (
                pl.when(pl.col(order_by).is_null()).then(non_null_count + null_count_before).otherwise(non_null_rank)
            )
            expr = ((row_num - 1) * ntile_n // group_size + 1).cast(pl.Int64).alias(feature_name)
        else:
            raise ValueError(f"Unsupported rank type: {rank_type}")

        result = data.with_columns(expr)
        return result.drop("__mloda_null_flag")
