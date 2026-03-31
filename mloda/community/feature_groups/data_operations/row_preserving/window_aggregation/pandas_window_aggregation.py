"""Pandas implementation for window aggregation feature groups."""

from __future__ import annotations

from typing import Optional, Set, Type, Union

import pandas as pd

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
    WindowAggregationFeatureGroup,
)
from mloda.community.feature_groups.data_operations.pandas_helpers import (
    PANDAS_AGG_FUNCS,
    apply_null_safe_agg,
    coerce_count_dtype,
    null_safe_groupby,
)


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
        pandas_func = PANDAS_AGG_FUNCS.get(agg_type)
        if pandas_func is None:
            raise ValueError(f"Unsupported aggregation type: {agg_type}")

        grouped = null_safe_groupby(data, partition_by, source_col)
        result_series = apply_null_safe_agg(grouped, pandas_func, agg_type, method="transform")

        data = data.copy()
        data[feature_name] = result_series

        coerce_count_dtype(data, feature_name, agg_type)

        return data
