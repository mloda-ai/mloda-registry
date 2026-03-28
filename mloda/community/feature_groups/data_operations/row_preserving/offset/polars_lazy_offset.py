"""Polars Lazy implementation for offset feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

import polars as pl

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.offset.base import (
    OffsetFeatureGroup,
)


class PolarsLazyOffset(OffsetFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PolarsLazyDataFrame}

    @classmethod
    def _compute_offset(
        cls,
        data: pl.LazyFrame,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        order_by: str,
        offset_type: str,
    ) -> pl.LazyFrame:
        """Compute offset using Polars expressions (fully lazy).

        We add a row index, sort within partitions, compute the offset,
        then restore original row order.
        """
        # Track original row order
        data = data.with_row_index("__mloda_orig_idx")

        # Sort by partition_by + order_by (nulls last) for correct offset
        data = data.sort(partition_by + [order_by], nulls_last=True)

        if offset_type.startswith("lag_"):
            offset_n = int(offset_type[len("lag_") :])
            expr = pl.col(source_col).shift(offset_n).over(partition_by).alias(feature_name)
        elif offset_type.startswith("lead_"):
            offset_n = int(offset_type[len("lead_") :])
            expr = pl.col(source_col).shift(-offset_n).over(partition_by).alias(feature_name)
        elif offset_type.startswith("diff_"):
            offset_n = int(offset_type[len("diff_") :])
            expr = (pl.col(source_col) - pl.col(source_col).shift(offset_n)).over(partition_by).alias(feature_name)
        elif offset_type.startswith("pct_change_"):
            offset_n = int(offset_type[len("pct_change_") :])
            prev = pl.col(source_col).shift(offset_n)
            expr = (
                pl.when(prev.is_not_null() & (prev != 0))
                .then((pl.col(source_col) - prev) / prev)
                .otherwise(pl.lit(None))
                .over(partition_by)
                .alias(feature_name)
            )
        elif offset_type == "first_value":
            expr = pl.col(source_col).drop_nulls().first().over(partition_by).alias(feature_name)
        elif offset_type == "last_value":
            expr = pl.col(source_col).drop_nulls().last().over(partition_by).alias(feature_name)
        else:
            raise ValueError(f"Unsupported offset type: {offset_type}")

        result = data.with_columns(expr)
        # Restore original row order and drop helper column
        return result.sort("__mloda_orig_idx").drop("__mloda_orig_idx")
