"""Pandas implementation for percentile feature groups."""

from __future__ import annotations

from typing import Any

import pandas as pd
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.mask_utils import apply_pandas_mask
from mloda.community.feature_groups.data_operations.row_preserving.percentile.base import (
    PercentileFeatureGroup,
)


class PandasPercentile(Implementation[PercentileFeatureGroup]):
    feature_group = PercentileFeatureGroup
    compute_framework = PandasDataFrame

    @classmethod
    def calculate(
        cls, group: type[PercentileFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.percentile(data, features, compute_percentile=cls._compute_percentile)

    @classmethod
    def _compute_percentile(
        cls,
        data: pd.DataFrame,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        percentile: float,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> pd.DataFrame:
        agg_source = source_col
        if mask_spec is not None:
            data, agg_source = apply_pandas_mask(data, source_col, mask_spec, taken={feature_name})
        else:
            data = data.copy()
        by: str | list[str] = partition_by[0] if len(partition_by) == 1 else partition_by
        grouped = data.groupby(by, dropna=False)[agg_source]
        data[feature_name] = grouped.transform(lambda x: x.quantile(percentile))
        if mask_spec is not None:
            data = data.drop(columns=[agg_source])
        return data
