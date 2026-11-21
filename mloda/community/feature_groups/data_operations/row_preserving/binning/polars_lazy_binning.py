"""Polars Lazy implementation for binning feature groups."""

from __future__ import annotations

from typing import Any

import polars as pl
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.binning.base import (
    BinningFeatureGroup,
)

# Integer dtypes that always fit exactly in a float64.
_NARROW_INTEGERS = (pl.Int8, pl.Int16, pl.Int32, pl.UInt8, pl.UInt16, pl.UInt32)


def _non_numeric_descriptor(data: pl.LazyFrame, source_col: str) -> Any | None:
    """Binning accepts integer or floating only (decimal is rejected, unlike arithmetic)."""
    dtype = data.collect_schema()[source_col]
    if dtype == pl.Boolean or not (dtype.is_integer() or dtype.is_float()):
        return dtype
    return None


class PolarsLazyBinning(Implementation[BinningFeatureGroup]):
    feature_group = BinningFeatureGroup
    compute_framework = PolarsLazyDataFrame

    @classmethod
    def calculate(
        cls, group: type[BinningFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.assign_bins(
            data,
            features,
            compute_binning=cls._compute_binning,
            non_numeric=_non_numeric_descriptor,
            integer_bounds=cls._integer_bounds,
        )

    @classmethod
    def _integer_bounds(cls, data: pl.LazyFrame, source_col: str) -> tuple[int, int] | None:
        dtype = data.collect_schema()[source_col]
        if not dtype.is_integer() or dtype in _NARROW_INTEGERS:
            return None
        row = data.select(pl.col(source_col).min().alias("lo"), pl.col(source_col).max().alias("hi")).collect().row(0)
        return None if row[0] is None else (int(row[0]), int(row[1]))

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
        col = pl.col(source_col).cast(pl.Float64).fill_nan(None)
        col_min = col.min()
        col_max = col.max()
        col_range = col_max - col_min

        safe_width = pl.when(col_range == 0).then(pl.lit(1.0)).otherwise(col_range / n_bins)

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
