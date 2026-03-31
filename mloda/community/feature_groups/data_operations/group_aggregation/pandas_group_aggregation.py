"""Pandas implementation for group aggregation feature groups."""

from __future__ import annotations

from typing import Set, Type, Union

import pandas as pd

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.group_aggregation.base import (
    GroupAggregationFeatureGroup,
)
from mloda.community.feature_groups.data_operations.pandas_helpers import (
    PANDAS_AGG_FUNCS,
    apply_null_safe_agg,
    coerce_count_dtype,
    null_safe_groupby,
)


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
        pandas_func = PANDAS_AGG_FUNCS.get(agg_type)
        if pandas_func is None:
            raise ValueError(f"Unsupported aggregation type: {agg_type}")

        grouped = null_safe_groupby(data, partition_by, source_col)
        result = apply_null_safe_agg(grouped, pandas_func, agg_type).reset_index()
        result = result.rename(columns={source_col: feature_name})

        coerce_count_dtype(result, feature_name, agg_type)

        return result
