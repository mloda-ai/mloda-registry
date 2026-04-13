"""Pandas implementation for single-column global aggregate broadcast."""

from __future__ import annotations

from typing import Any

import pandas as pd

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error
from mloda.community.feature_groups.data_operations.mask_utils import build_mask_from_spec
from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.base import (
    ScalarAggregateFeatureGroup,
)
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_mask_engine import (
    PandasMaskEngine,
)

_SUPPORTED_AGG_TYPES = {
    "sum",
    "min",
    "max",
    "avg",
    "mean",
    "count",
    "std",
    "std_pop",
    "std_samp",
    "var",
    "var_pop",
    "var_samp",
    "median",
}


class PandasScalarAggregate(ScalarAggregateFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame}

    @classmethod
    def _compute_aggregation(
        cls,
        data: pd.DataFrame,
        feature_name: str,
        source_col: str,
        agg_type: str,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> pd.DataFrame:
        if mask_spec is not None:
            mask = build_mask_from_spec(PandasMaskEngine, data, mask_spec)
            data = data.copy()
            data[source_col] = data[source_col].where(mask)
        else:
            data = data.copy()
        col = data[source_col]

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
            result = col.median()
        else:
            raise unsupported_agg_type_error(agg_type, _SUPPORTED_AGG_TYPES, framework="Pandas")

        data[feature_name] = result
        return data
