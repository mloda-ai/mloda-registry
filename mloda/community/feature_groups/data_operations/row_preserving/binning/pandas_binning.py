"""Pandas implementation for binning feature groups."""

from __future__ import annotations

from typing import Set, Type, Union

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
        non_null = col.dropna()

        if len(non_null) == 0 or non_null.nunique() <= 1:
            data[feature_name] = col.apply(lambda v: None if pd.isna(v) else 0)
            return data

        if op == "bin":
            binned = pd.cut(col, bins=n_bins, labels=False)
        elif op == "qbin":
            binned = pd.qcut(col, q=n_bins, labels=False, duplicates="drop")
        else:
            raise ValueError(f"Unsupported binning operation: {op}")

        data[feature_name] = binned
        return data
