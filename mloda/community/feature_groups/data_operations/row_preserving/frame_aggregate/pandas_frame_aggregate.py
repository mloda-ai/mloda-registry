"""Pandas implementation for frame aggregate feature groups."""

from __future__ import annotations

from typing import Optional, Set, Type, Union

import pandas as pd

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.base import (
    FrameAggregateFeatureGroup,
)
from mloda.community.feature_groups.data_operations.pandas_helpers import (
    PANDAS_AGG_FUNCS,
    coerce_count_dtype,
    null_safe_groupby,
)

_PANDAS_FRAME_AGG_FUNCS: dict[str, str] = {
    **PANDAS_AGG_FUNCS,
    "std": "std",
    "var": "var",
    "median": "median",
}


class PandasFrameAggregate(FrameAggregateFeatureGroup):
    SUPPORTED_FRAME_TYPES = {"rolling", "cumulative", "expanding"}

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
        pandas_func = _PANDAS_FRAME_AGG_FUNCS.get(agg_type)
        if pandas_func is None:
            raise ValueError(f"Unsupported aggregation type for Pandas frame aggregate: {agg_type}")

        data = data.copy()

        # Save original row order so we can restore it after sorting
        rn_col = "__mloda_rn__"
        data[rn_col] = range(len(data))

        data = data.sort_values(by=[*partition_by, order_by], na_position="last")

        grouped = null_safe_groupby(data, partition_by, source_col)

        # std/var require at least 2 observations for a meaningful result
        min_periods = 2 if agg_type in ("std", "var") else 1
        reset_levels = list(range(len(partition_by)))

        if frame_type in ("cumulative", "expanding"):
            window_obj = grouped.expanding(min_periods=min_periods)
        elif frame_type == "rolling":
            window = int(frame_size) if frame_size is not None else 1
            window_obj = grouped.rolling(window=window, min_periods=min_periods)
        else:
            raise ValueError(f"Unsupported frame type for Pandas: {frame_type}")

        if agg_type in ("std", "var"):
            result = getattr(window_obj, pandas_func)(ddof=0).reset_index(level=reset_levels, drop=True)
        else:
            result = getattr(window_obj, pandas_func)().reset_index(level=reset_levels, drop=True)

        data[feature_name] = result
        coerce_count_dtype(data, feature_name, agg_type)

        # Restore original row order and drop helper column
        data = data.sort_values(by=rn_col)
        data = data.drop(columns=[rn_col])

        return data
