"""Pandas implementation for string operation feature groups."""

from __future__ import annotations

from typing import Set, Type, Union

import pandas as pd

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.string.base import (
    StringFeatureGroup,
)


class PandasStringOps(StringFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PandasDataFrame}

    @classmethod
    def _compute_string(
        cls,
        data: pd.DataFrame,
        feature_name: str,
        source_col: str,
        op: str,
    ) -> pd.DataFrame:
        data = data.copy()
        col = data[source_col]

        if op == "upper":
            data[feature_name] = col.str.upper()
        elif op == "lower":
            data[feature_name] = col.str.lower()
        elif op == "trim":
            data[feature_name] = col.str.strip()
        elif op == "length":
            data[feature_name] = col.str.len()
        elif op == "reverse":
            data[feature_name] = col.str[::-1]
        else:
            raise ValueError(f"Unsupported string operation: {op}")

        return data
