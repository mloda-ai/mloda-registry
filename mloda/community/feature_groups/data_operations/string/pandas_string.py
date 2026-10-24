"""Pandas implementation for string operation feature groups."""

from __future__ import annotations

from typing import Any

import pandas as pd
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.string.base import (
    StringFeatureGroup,
)


class PandasStringOps(Implementation[StringFeatureGroup]):
    feature_group = StringFeatureGroup
    compute_framework = PandasDataFrame

    @classmethod
    def calculate(
        cls, group: type[StringFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.transform_string(data, features, compute_string=cls._compute_string)

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
