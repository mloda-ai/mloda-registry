"""Pandas implementation for window aggregation feature groups."""

from __future__ import annotations

from typing import Any, Optional, Set, Type, Union

import pandas as pd

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
    WindowAggregationFeatureGroup,
)

# Mapping from aggregation type to the pandas GroupBy.transform function name.
_PANDAS_AGG_FUNCS: dict[str, str] = {
    "sum": "sum",
    "avg": "mean",
    "count": "count",
    "min": "min",
    "max": "max",
}


class PandasWindowAggregation(WindowAggregationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PandasDataFrame}

    @classmethod
    def _compute_window(
        cls,
        data: pd.DataFrame,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
        order_by: Optional[str] = None,
    ) -> pd.DataFrame:
        """Compute a window aggregation using pandas groupby().transform()."""
        pandas_func = _PANDAS_AGG_FUNCS.get(agg_type)
        if pandas_func is None:
            raise ValueError(f"Unsupported aggregation type: {agg_type}")

        # CRITICAL: dropna=False ensures null group keys form their own group,
        # matching PyArrow behavior. Without this, pandas drops null keys entirely.
        # min_count=1 for sum ensures all-null groups return NaN (not 0), matching PyArrow.
        if agg_type == "sum":
            result_series = data.groupby(partition_by, dropna=False)[source_col].transform(pandas_func, min_count=1)
        else:
            result_series = data.groupby(partition_by, dropna=False)[source_col].transform(pandas_func)

        data = data.copy()
        data[feature_name] = result_series

        # Convert count results to int64 (pandas transform may produce float when nulls exist)
        if agg_type == "count":
            data[feature_name] = data[feature_name].astype("int64")

        return data
