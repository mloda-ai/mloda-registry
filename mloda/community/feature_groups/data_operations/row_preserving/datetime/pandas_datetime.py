"""Pandas implementation for datetime extraction feature groups."""

from __future__ import annotations

from typing import Set, Type, Union

import pandas as pd

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.datetime.base import (
    DateTimeFeatureGroup,
)


class PandasDateTimeExtraction(DateTimeFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PandasDataFrame}

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

        if op == "year":
            data[feature_name] = col.dt.year
        elif op == "month":
            data[feature_name] = col.dt.month
        elif op == "day":
            data[feature_name] = col.dt.day
        elif op == "hour":
            data[feature_name] = col.dt.hour
        elif op == "minute":
            data[feature_name] = col.dt.minute
        elif op == "second":
            data[feature_name] = col.dt.second
        elif op == "dayofweek":
            data[feature_name] = col.dt.dayofweek
        elif op == "is_weekend":
            mask = col.notna()
            result = pd.array([pd.NA] * len(col), dtype="Int64")
            result[mask] = (col[mask].dt.dayofweek >= 5).astype(int).values
            data[feature_name] = result
        elif op == "quarter":
            data[feature_name] = col.dt.quarter
        else:
            raise ValueError(f"Unsupported datetime operation: {op}")

        return data
