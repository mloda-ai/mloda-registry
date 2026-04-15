"""Pandas implementation for window aggregation feature groups."""

from __future__ import annotations

from typing import Any

import pandas as pd

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error
from mloda.community.feature_groups.data_operations.mask_utils import build_mask_from_spec
from mloda.community.feature_groups.data_operations.reserved_columns import assert_no_reserved_columns
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
    WindowAggregationFeatureGroup,
)
from mloda.community.feature_groups.data_operations.pandas_helpers import (
    PANDAS_AGG_FUNCS,
    _unique_temp_name,
    apply_null_safe_agg,
    coerce_count_dtype,
    compute_mode_winners,
    null_safe_groupby,
)
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_mask_engine import (
    PandasMaskEngine,
)

_SUPPORTED_AGG_TYPES = {*PANDAS_AGG_FUNCS.keys(), "mode"}


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
        assert_no_reserved_columns(data.columns, framework="Pandas", operation="window aggregation")

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
            raise unsupported_agg_type_error(agg_type, _SUPPORTED_AGG_TYPES, framework="Pandas")

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
        partition_by: list[str] | tuple[str, ...],
    ) -> pd.DataFrame:
        """Insertion-order tie-breaking for PyArrow parity."""
        partition_by = list(partition_by)
        if source_col in partition_by:
            data = data.copy()
            data[feature_name] = data[source_col]
            return data

        is_data_col = _unique_temp_name("__mloda_mode_is_data__", data.columns)

        winners = compute_mode_winners(data, source_col, partition_by)
        winners = winners.rename(columns={source_col: feature_name})

        carrier = data[partition_by].copy()
        carrier[feature_name] = pd.NA
        carrier[is_data_col] = True

        winners_for_merge = winners.copy()
        winners_for_merge[is_data_col] = False

        combined = pd.concat([winners_for_merge, carrier], ignore_index=True, sort=False)
        combined[feature_name] = combined.groupby(partition_by, dropna=False)[feature_name].transform("first")

        broadcast = combined.loc[combined[is_data_col], feature_name]

        data = data.copy()
        data[feature_name] = broadcast.to_numpy()
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
        """PyArrow parity: sort within each partition for first/last semantics,
        then restore input row order via sort_index()."""
        pandas_func = PANDAS_AGG_FUNCS[agg_type]
        sorted_data = data.sort_values(order_by, na_position="last")
        grouped = null_safe_groupby(sorted_data, partition_by, source_col)
        result_series = grouped.transform(pandas_func)

        data = data.copy()
        data[feature_name] = result_series.sort_index()
        return data
