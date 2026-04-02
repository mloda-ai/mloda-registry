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
        if op == "bin":
            expr = cls._build_bin_expr(source_col, n_bins)
        elif op == "qbin":
            expr = cls._build_qbin_expr(source_col, n_bins)
        else:
            raise ValueError(f"Unsupported binning operation: {op}")

        return data.with_columns(expr.alias(feature_name))

    @classmethod
    def _build_bin_expr(cls, source_col: str, n_bins: int) -> pl.Expr:
        col = pl.col(source_col).fill_nan(None)
        col_min = col.min()
        col_max = col.max()
        col_range = col_max - col_min

        safe_width = pl.when(col_range == 0).then(pl.lit(1.0)).otherwise(col_range.cast(pl.Float64) / n_bins)

        raw_bin = ((col - col_min) / safe_width).floor()

        clamped = pl.when(raw_bin >= n_bins).then(pl.lit(n_bins - 1).cast(pl.Float64)).otherwise(raw_bin)

        return (
            pl.when(col.is_null())
            .then(pl.lit(None, dtype=pl.Int64))
            .when(col_min == col_max)
            .then(pl.lit(0, dtype=pl.Int64))
            .otherwise(clamped.cast(pl.Int64))
        )

    @classmethod
    def _build_qbin_expr(cls, source_col: str, n_bins: int) -> pl.Expr:
        col = pl.col(source_col).fill_nan(None)
        rank_expr = col.rank("ordinal") - 1
        count_expr = col.count()

        bin_expr = (rank_expr * n_bins) // count_expr

        return pl.when(col.is_null()).then(pl.lit(None, dtype=pl.Int64)).otherwise(bin_expr).cast(pl.Int64)
