"""Pandas implementation for binning feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

import pandas as pd

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.binning.base import (
    BinningFeatureGroup,
)


class PandasBinning(BinningFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PandasDataFrame}

    @classmethod
    def _compute_binning(
        cls,
        data: pd.DataFrame,
        feature_name: str,
        source_col: str,
        op: str,
        n_bins: int,
    ) -> pd.DataFrame:
        data = data.copy()
        col = data[source_col]
        non_null_mask = col.notna()

        if non_null_mask.sum() == 0:
            data[feature_name] = None
            return data

        if op == "bin":
            if col[non_null_mask].nunique() <= 1:
                data[feature_name] = col.apply(lambda v: None if pd.isna(v) else 0)
                return data
            binned = pd.cut(col, bins=n_bins, labels=False)
            data[feature_name] = binned
        elif op == "qbin":
            data[feature_name] = cls._rank_based_qbin(col, n_bins)
        else:
            raise ValueError(f"Unsupported binning operation: {op}")

        return data

    @classmethod
    def _rank_based_qbin(cls, col: "pd.Series[Any]", n_bins: int) -> "pd.Series[Any]":
        """Rank-based quantile binning matching NTILE semantics.

        Rows are sorted by value and divided into n_bins roughly equal groups.
        For N non-null values, rank r (0-based) maps to bin = r * n_bins // N.
        """
        values = col.tolist()
        indexed = [(v, i) for i, v in enumerate(values) if not pd.isna(v)]
        indexed.sort(key=lambda pair: pair[0])
        n = len(indexed)

        result: list[Any] = [None] * len(values)
        for rank, (_, orig_idx) in enumerate(indexed):
            result[orig_idx] = rank * n_bins // n

        return pd.Series(result, dtype="Int64")
