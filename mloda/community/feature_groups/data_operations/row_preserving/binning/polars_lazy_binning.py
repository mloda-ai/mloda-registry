"""Polars Lazy implementation for binning feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

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
        collected = data.collect()
        col = collected[source_col]
        values = col.to_list()
        non_null = [v for v in values if v is not None]

        if not non_null:
            result_values: list[Any] = [None] * len(values)
            result_series = pl.Series(feature_name, result_values, dtype=pl.Int64)
            return collected.with_columns(result_series).lazy()

        if op == "bin":
            result_values = cls._equal_width_binning(values, non_null, n_bins)
        elif op == "qbin":
            result_values = cls._quantile_binning(values, non_null, n_bins)
        else:
            raise ValueError(f"Unsupported binning operation: {op}")

        result_series = pl.Series(feature_name, result_values, dtype=pl.Int64)
        return collected.with_columns(result_series).lazy()

    @classmethod
    def _equal_width_binning(cls, values: list[Any], non_null: list[Any], n_bins: int) -> list[Any]:
        min_val = min(non_null)
        max_val = max(non_null)

        result: list[Any] = []
        for val in values:
            if val is None:
                result.append(None)
                continue
            if min_val == max_val:
                result.append(0)
                continue
            bin_width = (max_val - min_val) / n_bins
            bin_idx = int((val - min_val) / bin_width)
            if bin_idx >= n_bins:
                bin_idx = n_bins - 1
            result.append(bin_idx)
        return result

    @classmethod
    def _quantile_binning(cls, values: list[Any], non_null: list[Any], n_bins: int) -> list[Any]:
        """Rank-based quantile binning matching NTILE semantics.

        Rows are sorted by value and divided into n_bins roughly equal groups.
        For N non-null values, rank r (0-based) maps to bin = r * n_bins // N.
        Ties receive consecutive ranks (same value may span two bins at a boundary).
        """
        indexed = [(v, i) for i, v in enumerate(values) if v is not None]
        indexed.sort(key=lambda pair: pair[0])
        n = len(indexed)

        result: list[Any] = [None] * len(values)
        for rank, (_, orig_idx) in enumerate(indexed):
            result[orig_idx] = rank * n_bins // n

        return result
