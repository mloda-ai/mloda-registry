"""Pandas implementation for percentile feature groups."""

from __future__ import annotations

from typing import Any

import numpy as np
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
        values = data[agg_source].astype("float64")
        grouped = values.groupby([data[c] for c in partition_by], dropna=False)
        lo = grouped.transform("quantile", percentile, interpolation="lower")
        hi = grouped.transform("quantile", percentile, interpolation="higher")
        lin = grouped.transform("quantile", percentile, interpolation="linear")
        # Linear interpolation next to an infinity can give NaN; the reference gives the infinite neighbour (lo + hi).
        either_inf = np.isinf(lo) | np.isinf(hi)
        data[feature_name] = np.where(lo == hi, lo, np.where(either_inf, lo + hi, lin))
        if mask_spec is not None:
            data = data.drop(columns=[agg_source])
        return data
