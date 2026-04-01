"""Pandas implementation for single-column global aggregate broadcast."""

from __future__ import annotations

from typing import Set, Type, Union

import pandas as pd

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.base import (
    ScalarAggregateFeatureGroup,
)


class PandasScalarAggregate(ScalarAggregateFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PandasDataFrame}

    @classmethod
    def _compute_aggregation(
        cls,
        data: pd.DataFrame,
        feature_name: str,
        source_col: str,
        agg_type: str,
    ) -> pd.DataFrame:
        data = data.copy()
        col = data[source_col]

        if agg_type == "sum":
            result = col.sum()
        elif agg_type == "min":
            result = col.min()
        elif agg_type == "max":
            result = col.max()
        elif agg_type in ("avg", "mean"):
            result = col.mean()
        elif agg_type == "count":
            result = col.count()
        elif agg_type == "std":
            result = col.std(ddof=0)
        elif agg_type == "var":
            result = col.var(ddof=0)
        elif agg_type == "median":
            result = col.median()
        else:
            raise ValueError(f"Unsupported aggregation type: {agg_type}")

        data[feature_name] = result
        return data
