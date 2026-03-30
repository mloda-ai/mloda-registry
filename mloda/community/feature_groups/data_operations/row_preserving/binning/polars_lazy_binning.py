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
            result_values = cls._quantile_binning(values, n_bins)
        else:
            raise ValueError(f"Unsupported binning operation: {op}")

        result_series = pl.Series(feature_name, result_values, dtype=pl.Int64)
        return collected.with_columns(result_series).lazy()
