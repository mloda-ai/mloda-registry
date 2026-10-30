"""Pandas implementation for datetime extraction feature groups."""

from __future__ import annotations

from typing import Any

import pandas as pd
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.datetime.base import (
    DateTimeFeatureGroup,
)

_ACCESSOR_OPS = ("year", "month", "day", "hour", "minute", "second", "dayofweek", "quarter")


class PandasDateTimeExtraction(Implementation[DateTimeFeatureGroup]):
    feature_group = DateTimeFeatureGroup
    compute_framework = PandasDataFrame

    @classmethod
    def calculate(
        cls, group: type[DateTimeFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.extract_datetime(data, features, compute_datetime=cls._compute_datetime)

    @classmethod
    def _compute_datetime(
        cls,
        data: pd.DataFrame,
        feature_name: str,
        source_col: str,
        op: str,
    ) -> pd.DataFrame:
        data = data.copy()
        col = pd.to_datetime(data[source_col])

        if op == "is_weekend":
            data[feature_name] = (col.dt.day_of_week.astype("Int64") >= 5).astype("Int64")
        elif op in _ACCESSOR_OPS:
            accessor = "day_of_week" if op == "dayofweek" else op
            data[feature_name] = getattr(col.dt, accessor).astype("Int64")
        else:
            raise ValueError(f"Unsupported datetime operation: {op}")

        return data
