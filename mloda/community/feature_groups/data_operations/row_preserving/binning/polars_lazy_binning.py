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
        sorted_vals = sorted(non_null)
        n = len(sorted_vals)

        edges = []
        for i in range(n_bins + 1):
            pos = i * (n - 1) / n_bins
            lower_idx = int(pos)
            frac = pos - lower_idx
            if lower_idx + 1 < n:
                edge = sorted_vals[lower_idx] * (1 - frac) + sorted_vals[lower_idx + 1] * frac
            else:
                edge = sorted_vals[lower_idx]
            edges.append(edge)

        result: list[Any] = []
        for val in values:
            if val is None:
                result.append(None)
                continue
            bin_idx = 0
            for i in range(1, len(edges)):
                if val > edges[i]:
                    bin_idx = i
                else:
                    bin_idx = i - 1
                    break
            else:
                bin_idx = n_bins - 1
            if bin_idx >= n_bins:
                bin_idx = n_bins - 1
            result.append(bin_idx)
        return result
