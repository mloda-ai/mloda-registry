"""Pandas implementation for window aggregation feature groups."""

from __future__ import annotations

from typing import Any

import pandas as pd

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.mask_utils import build_mask_from_spec
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
    WindowAggregationFeatureGroup,
)
from mloda.community.feature_groups.data_operations.pandas_helpers import (
    PANDAS_AGG_FUNCS,
    apply_null_safe_agg,
    coerce_count_dtype,
    null_safe_groupby,
)
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_mask_engine import (
    PandasMaskEngine,
)


class PandasWindowAggregation(WindowAggregationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame}

    @classmethod
    def _compute_window(
        cls,
        data: pd.DataFrame,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
        order_by: str | None = None,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> pd.DataFrame:
        """Compute a window aggregation using pandas groupby().transform()."""
        if mask_spec is not None:
            mask = build_mask_from_spec(PandasMaskEngine, data, mask_spec)
            data = data.copy()
            data[source_col] = data[source_col].where(mask)

        if agg_type == "mode":
            return cls._compute_mode(data, feature_name, source_col, partition_by)

        if agg_type in ("first", "last") and order_by is not None:
            return cls._compute_ordered(data, feature_name, source_col, partition_by, agg_type, order_by)

        pandas_func = PANDAS_AGG_FUNCS.get(agg_type)
        if pandas_func is None:
            raise ValueError(f"Unsupported aggregation type: {agg_type}")

        grouped = null_safe_groupby(data, partition_by, source_col)
        result_series = apply_null_safe_agg(grouped, pandas_func, agg_type, method="transform")

        data = data.copy()
        data[feature_name] = result_series

        coerce_count_dtype(data, feature_name, agg_type)

        return data

    @classmethod
    def _compute_mode(
        cls,
        data: pd.DataFrame,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
    ) -> pd.DataFrame:
        """Compute mode via lambda because pandas has no string-based mode transform."""
        grouped = null_safe_groupby(data, partition_by, source_col)
        result_series = grouped.transform(lambda x: x.mode().iloc[0] if not x.mode().empty else None)

        data = data.copy()
        data[feature_name] = result_series
        return data

    @classmethod
    def _compute_ordered(
        cls,
        data: pd.DataFrame,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
        order_by: str,
    ) -> pd.DataFrame:
        """Compute first/last with order_by by sorting, transforming, then restoring row order."""
        pandas_func = PANDAS_AGG_FUNCS[agg_type]
        sorted_data = data.sort_values(order_by, na_position="last")
        grouped = null_safe_groupby(sorted_data, partition_by, source_col)
        result_series = grouped.transform(pandas_func)

        data = data.copy()
        data[feature_name] = result_series.sort_index()
        return data
