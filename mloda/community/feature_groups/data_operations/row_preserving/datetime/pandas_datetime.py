"""Pandas implementation for datetime extraction feature groups."""

from __future__ import annotations

import pandas as pd
from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.datetime.base import (
    DateTimeFeatureGroup,
)

_ACCESSOR_OPS = ("year", "month", "day", "hour", "minute", "second", "dayofweek", "quarter")


class PandasDateTimeExtraction(DateTimeFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
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

        if op == "is_weekend":
            data[feature_name] = (col.dt.dayofweek.astype("Int64") >= 5).astype("Int64")
        elif op in _ACCESSOR_OPS:
            data[feature_name] = getattr(col.dt, op).astype("Int64")
        else:
            raise ValueError(f"Unsupported datetime operation: {op}")

        return data
