"""Pandas implementation for aggregation feature groups."""

from __future__ import annotations

from typing import Any

import pandas as pd

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.aggregation.base import (
    AggregationFeatureGroup,
)
from mloda.community.feature_groups.data_operations.mask_utils import build_mask_from_spec
from mloda.community.feature_groups.data_operations.pandas_helpers import (
    PANDAS_AGG_FUNCS,
    apply_null_safe_agg,
    coerce_count_dtype,
    null_safe_groupby,
)
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_mask_engine import (
    PandasMaskEngine,
)


class PandasAggregation(AggregationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame}

    @classmethod
    def _compute_group(
        cls,
        data: pd.DataFrame,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> pd.DataFrame:
        """Compute a group aggregation using pandas groupby().agg()."""
        if mask_spec is not None:
            mask = build_mask_from_spec(PandasMaskEngine, data, mask_spec)
            data = data.copy()
            data[source_col] = data[source_col].where(mask)

        if agg_type == "mode":
            return cls._compute_mode(data, feature_name, source_col, partition_by)

        pandas_func = PANDAS_AGG_FUNCS.get(agg_type)
        if pandas_func is None:
            raise ValueError(f"Unsupported aggregation type: {agg_type}")

        grouped = null_safe_groupby(data, partition_by, source_col)
        result = apply_null_safe_agg(grouped, pandas_func, agg_type).reset_index()
        result = result.rename(columns={source_col: feature_name})

        coerce_count_dtype(result, feature_name, agg_type)

        return result

    @classmethod
    def _compute_mode(
        cls,
        data: pd.DataFrame,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
    ) -> pd.DataFrame:
        """Compute mode via lambda because pandas has no string-based mode aggregation."""
        grouped = null_safe_groupby(data, partition_by, source_col)
        result = grouped.agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None).reset_index()
        return result.rename(columns={source_col: feature_name})
