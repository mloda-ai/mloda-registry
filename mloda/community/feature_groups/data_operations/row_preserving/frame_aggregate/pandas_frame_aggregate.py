"""Pandas implementation for frame aggregate feature groups."""

from __future__ import annotations

from typing import Any, Optional, Set, Type, Union

import pandas as pd

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.base import (
    FrameAggregateFeatureGroup,
)

_PANDAS_AGG_FUNCS: dict[str, str] = {
    "sum": "sum",
    "avg": "mean",
    "count": "count",
    "min": "min",
    "max": "max",
    "std": "std",
    "var": "var",
    "median": "median",
}


class PandasFrameAggregate(FrameAggregateFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PandasDataFrame}

    @classmethod
    def _compute_frame(
        cls,
        data: pd.DataFrame,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        order_by: str,
        agg_type: str,
        frame_type: str,
        frame_size: Optional[int] = None,
        frame_unit: Optional[str] = None,
    ) -> pd.DataFrame:
        pandas_func = _PANDAS_AGG_FUNCS.get(agg_type)
        if pandas_func is None:
            raise ValueError(f"Unsupported aggregation type for Pandas frame aggregate: {agg_type}")

        data = data.copy()
        data = data.sort_values(by=[*partition_by, order_by], na_position="last")

        grouped = data.groupby(partition_by, dropna=False)[source_col]

        if frame_type in ("cumulative", "expanding"):
            if agg_type == "sum":
                result = (
                    grouped.expanding(min_periods=1).sum().reset_index(level=list(range(len(partition_by))), drop=True)
                )
            elif agg_type == "count":
                result = (
                    grouped.expanding(min_periods=1)
                    .count()
                    .reset_index(level=list(range(len(partition_by))), drop=True)
                )
            elif agg_type == "min":
                result = (
                    grouped.expanding(min_periods=1).min().reset_index(level=list(range(len(partition_by))), drop=True)
                )
            elif agg_type == "max":
                result = (
                    grouped.expanding(min_periods=1).max().reset_index(level=list(range(len(partition_by))), drop=True)
                )
            elif agg_type == "avg":
                result = (
                    grouped.expanding(min_periods=1).mean().reset_index(level=list(range(len(partition_by))), drop=True)
                )
            elif agg_type == "std":
                result = (
                    grouped.expanding(min_periods=2).std().reset_index(level=list(range(len(partition_by))), drop=True)
                )
            elif agg_type == "var":
                result = (
                    grouped.expanding(min_periods=2).var().reset_index(level=list(range(len(partition_by))), drop=True)
                )
            elif agg_type == "median":
                result = (
                    grouped.expanding(min_periods=1)
                    .median()
                    .reset_index(level=list(range(len(partition_by))), drop=True)
                )
            else:
                raise ValueError(f"Unsupported cumulative/expanding agg: {agg_type}")
        elif frame_type == "rolling":
            window = int(frame_size) if frame_size is not None else 1
            if agg_type == "sum":
                result = (
                    grouped.rolling(window=window, min_periods=1)
                    .sum()
                    .reset_index(level=list(range(len(partition_by))), drop=True)
                )
            elif agg_type == "count":
                result = (
                    grouped.rolling(window=window, min_periods=1)
                    .count()
                    .reset_index(level=list(range(len(partition_by))), drop=True)
                )
            elif agg_type == "min":
                result = (
                    grouped.rolling(window=window, min_periods=1)
                    .min()
                    .reset_index(level=list(range(len(partition_by))), drop=True)
                )
            elif agg_type == "max":
                result = (
                    grouped.rolling(window=window, min_periods=1)
                    .max()
                    .reset_index(level=list(range(len(partition_by))), drop=True)
                )
            elif agg_type == "avg":
                result = (
                    grouped.rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=list(range(len(partition_by))), drop=True)
                )
            elif agg_type == "std":
                result = (
                    grouped.rolling(window=window, min_periods=2)
                    .std()
                    .reset_index(level=list(range(len(partition_by))), drop=True)
                )
            elif agg_type == "var":
                result = (
                    grouped.rolling(window=window, min_periods=2)
                    .var()
                    .reset_index(level=list(range(len(partition_by))), drop=True)
                )
            elif agg_type == "median":
                result = (
                    grouped.rolling(window=window, min_periods=1)
                    .median()
                    .reset_index(level=list(range(len(partition_by))), drop=True)
                )
            else:
                raise ValueError(f"Unsupported rolling agg: {agg_type}")
        else:
            raise ValueError(f"Unsupported frame type for Pandas: {frame_type}")

        data[feature_name] = result
        if agg_type == "count":
            data[feature_name] = data[feature_name].astype("int64")

        return data
