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
            non_null_vals = col[non_null_mask].tolist()
            result = cls._equal_width_binning(col.tolist(), non_null_vals, n_bins)
            data[feature_name] = pd.Series(result, dtype="Int64")
        elif op == "qbin":
            result = cls._quantile_binning(col.tolist(), n_bins)
            data[feature_name] = pd.Series(result, dtype="Int64")
        else:
            raise ValueError(f"Unsupported binning operation: {op}")

        return data
