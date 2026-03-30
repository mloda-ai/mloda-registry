"""Pandas implementation for group aggregation feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

import pandas as pd

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.group_aggregation.base import (
    GroupAggregationFeatureGroup,
)

# Mapping from aggregation type to the pandas GroupBy aggregate function name.
_PANDAS_AGG_FUNCS: dict[str, str] = {
    "sum": "sum",
    "avg": "mean",
    "count": "count",
    "min": "min",
    "max": "max",
}


class PandasGroupAggregation(GroupAggregationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PandasDataFrame}

    @classmethod
    def _compute_group(
        cls,
        data: pd.DataFrame,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
    ) -> pd.DataFrame:
        """Compute a group aggregation using pandas groupby().agg()."""
        pandas_func = _PANDAS_AGG_FUNCS.get(agg_type)
        if pandas_func is None:
            raise ValueError(f"Unsupported aggregation type: {agg_type}")

        # CRITICAL: dropna=False ensures null group keys form their own group,
        # matching PyArrow behavior. Without this, pandas drops null keys entirely.
        # min_count=1 for sum ensures all-null groups return NaN (not 0), matching PyArrow.
        if agg_type == "sum":
            grouped = data.groupby(partition_by, dropna=False)[source_col].agg(pandas_func, min_count=1)
        else:
            grouped = data.groupby(partition_by, dropna=False)[source_col].agg(pandas_func)
        result = grouped.reset_index()
        result = result.rename(columns={source_col: feature_name})

        # Convert count results to int64 (pandas may produce float when nulls exist)
        if agg_type == "count":
            result[feature_name] = result[feature_name].astype("int64")

        return result
