"""Pandas implementation for single-column global aggregate broadcast."""

from __future__ import annotations

from typing import Any

import pandas as pd
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error
from mloda.community.feature_groups.data_operations.mask_utils import apply_pandas_mask
from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.base import (
    ScalarAggregateFeatureGroup,
)


class PandasScalarAggregate(Implementation[ScalarAggregateFeatureGroup]):
    feature_group = ScalarAggregateFeatureGroup
    compute_framework = PandasDataFrame
    divergent = {
        "sum": "skips NaN",
        "avg": "skips NaN",
        "mean": "skips NaN",
        "count": "does not count NaN",
        "std": "skips NaN",
        "var": "skips NaN",
        "std_pop": "skips NaN",
        "var_pop": "skips NaN",
        "std_samp": "skips NaN",
        "var_samp": "skips NaN",
    }

    @classmethod
    def calculate(
        cls,
        group: type[ScalarAggregateFeatureGroup],
        data: Any,
        features: FeatureSet,
        framework: ComputeFramework | None,
    ) -> Any:
        return group.aggregate(data, features, compute_aggregation=cls._compute_aggregation)

    @classmethod
    def _compute_aggregation(
        cls,
        data: pd.DataFrame,
        feature_name: str,
        source_col: str,
        agg_type: str,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> pd.DataFrame:
        agg_source = source_col
        if mask_spec is not None:
            data, agg_source = apply_pandas_mask(data, source_col, mask_spec, taken={feature_name})
        else:
            data = data.copy()
        col = data[agg_source]

        if agg_type == "sum":
            result = col.sum(min_count=1)
        elif agg_type == "min":
            result = col.min()
        elif agg_type == "max":
            result = col.max()
        elif agg_type in ("avg", "mean"):
            result = col.mean()
        elif agg_type == "count":
            result = col.count()
        elif agg_type in ("std", "std_pop"):
            result = col.std(ddof=0)
        elif agg_type in ("var", "var_pop"):
            result = col.var(ddof=0)
        elif agg_type == "std_samp":
            result = col.std(ddof=1)
        elif agg_type == "var_samp":
            result = col.var(ddof=1)
        elif agg_type == "median":
            if col.isna().all():
                result = float("nan")
            else:
                result = col.median()
        else:
            raise unsupported_agg_type_error(
                agg_type, ScalarAggregateFeatureGroup._SUPPORTED_AGG_TYPES, framework="Pandas"
            )

        data[feature_name] = result
        if mask_spec is not None:
            data = data.drop(columns=[agg_source])
        return data
