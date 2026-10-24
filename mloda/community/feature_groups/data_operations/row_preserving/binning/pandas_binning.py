"""Pandas implementation for binning feature groups."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.binning.base import (
    BinningFeatureGroup,
)


class PandasBinning(Implementation[BinningFeatureGroup]):
    feature_group = BinningFeatureGroup
    compute_framework = PandasDataFrame

    @classmethod
    def calculate(
        cls, group: type[BinningFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.assign_bins(data, features, compute_binning=cls._compute_binning)

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
            col_min = col[non_null_mask].min()
            col_max = col[non_null_mask].max()
            bin_width = (col_max - col_min) / n_bins

            if col_min == col_max:
                result = pd.array([0 if m else pd.NA for m in non_null_mask], dtype="Int64")
            else:
                bin_idx = np.floor((col - col_min) / bin_width)
                bin_idx = bin_idx.clip(upper=n_bins - 1)
                result = bin_idx.astype("Int64")

            data[feature_name] = result
        elif op == "qbin":
            n = non_null_mask.sum()
            # rank non-null rows only: pandas 2 ranks nullable Int64 nulls first
            rank = col[non_null_mask].rank(method="first").reindex(col.index) - 1
            result = (rank * n_bins // n).astype("Int64")
            data[feature_name] = result
        else:
            raise ValueError(f"Unsupported binning operation: {op}")

        return data
