"""Polars Lazy implementation for binning feature groups."""

from __future__ import annotations

from typing import Set, Type, Union

import polars as pl

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.binning.base import (
    BinningFeatureGroup,
)


class PolarsLazyBinning(BinningFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PolarsLazyDataFrame}

    @classmethod
    def _compute_binning(
        cls,
        data: pl.LazyFrame,
        feature_name: str,
        source_col: str,
        op: str,
        n_bins: int,
    ) -> pl.LazyFrame:
        col = pl.col(source_col)

        if op == "bin":
            col_min = col.min()
            col_max = col.max()
            bin_width = (col_max - col_min) / n_bins

            expr = (
                pl.when(col.is_null())
                .then(None)
                .when(col_max == col_min)
                .then(pl.lit(0))
                .otherwise(((col - col_min) / bin_width).fill_nan(0).floor().cast(pl.Int64).clip(0, n_bins - 1))
                .cast(pl.Int64)
                .alias(feature_name)
            )
        elif op == "qbin":
            non_null_count = col.count()
            rank = col.rank(method="ordinal")

            expr = (
                pl.when(col.is_null())
                .then(None)
                .otherwise(((rank - 1) * n_bins / non_null_count).floor().cast(pl.Int64).clip(0, n_bins - 1))
                .cast(pl.Int64)
                .alias(feature_name)
            )
        else:
            raise ValueError(f"Unsupported binning operation: {op}")

        return data.with_columns(expr)
