"""Pandas implementation for filtered aggregation feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

import pandas as pd

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.filtered_aggregation.base import (
    FilteredAggregationFeatureGroup,
)
from mloda.community.feature_groups.data_operations.pandas_helpers import (
    PANDAS_AGG_FUNCS,
    apply_null_safe_agg,
    coerce_count_dtype,
    null_safe_groupby,
)

_TEMP_COL = "__mloda_fa_masked__"


class PandasFilteredAggregation(FilteredAggregationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PandasDataFrame}

    @classmethod
    def _compute_filtered(
        cls,
        data: pd.DataFrame,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
        filter_column: str,
        filter_value: Any,
    ) -> pd.DataFrame:
        """Compute a filtered aggregation using pandas groupby().transform()."""
        pandas_func = PANDAS_AGG_FUNCS.get(agg_type)
        if pandas_func is None:
            raise ValueError(f"Unsupported aggregation type: {agg_type}")

        data = data.copy()

        # Mask source column: keep value only where filter matches, NaN otherwise.
        mask = data[filter_column] == filter_value
        data[_TEMP_COL] = data[source_col].where(mask)

        grouped = null_safe_groupby(data, partition_by, _TEMP_COL)
        result_series = apply_null_safe_agg(grouped, pandas_func, agg_type, method="transform")

        data[feature_name] = result_series
        coerce_count_dtype(data, feature_name, agg_type)
        data = data.drop(columns=[_TEMP_COL])

        return data
