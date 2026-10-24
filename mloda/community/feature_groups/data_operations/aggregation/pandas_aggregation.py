"""Pandas implementation for aggregation feature groups."""

from __future__ import annotations

from typing import Any

import pandas as pd
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_mask_engine import (
    PandasMaskEngine,
)

from mloda.community.feature_groups.data_operations.aggregation.base import (
    AggregationFeatureGroup,
    AggregationSpec,
)
from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error
from mloda.community.feature_groups.data_operations.helper_columns import unique_helper_name
from mloda.community.feature_groups.data_operations.mask_utils import build_mask_from_spec
from mloda.community.feature_groups.data_operations.pandas_helpers import (
    PANDAS_AGG_FUNCS,
    apply_null_safe_agg,
    coerce_count_dtype,
    compute_mode_winners,
    null_safe_groupby,
)

_SUPPORTED_AGG_TYPES = {*PANDAS_AGG_FUNCS.keys(), "mode"}


class PandasAggregation(Implementation[AggregationFeatureGroup]):
    feature_group = AggregationFeatureGroup
    compute_framework = PandasDataFrame
    divergent = {
        "sum": "skips NaN",
        "avg": "skips NaN",
        "mean": "skips NaN",
        "std": "skips NaN",
        "var": "skips NaN",
        "std_pop": "skips NaN",
        "var_pop": "skips NaN",
        "std_samp": "skips NaN",
        "var_samp": "skips NaN",
        "count": "does not count NaN",
        "mode": "does not count NaN as a value",
    }

    @classmethod
    def calculate(
        cls, group: type[AggregationFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.aggregate(data, features, compute_group=cls._compute_group)

    @classmethod
    def _compute_group(cls, data: pd.DataFrame, specs: list[AggregationSpec], partition_by: list[str]) -> pd.DataFrame:
        for spec in specs:
            if spec.agg_type != "mode" and spec.agg_type not in PANDAS_AGG_FUNCS:
                raise unsupported_agg_type_error(spec.agg_type, _SUPPORTED_AGG_TYPES, framework="Pandas")

        # Each spec is computed from the original data with its own mask, then the frames are combined.
        frames = [cls._compute_single(data, spec, partition_by) for spec in specs]
        result = frames[0]
        for frame in frames[1:]:
            if partition_by:
                # Every frame holds the same unique keys; pandas matches NaN keys to NaN keys.
                result = result.merge(frame, on=partition_by, how="left")
            else:
                result = pd.concat([result, frame], axis=1)
        return result

    @classmethod
    def _compute_single(cls, data: pd.DataFrame, spec: AggregationSpec, partition_by: list[str]) -> pd.DataFrame:
        feature_name, source_col, agg_type, mask_spec = spec
        if mask_spec is not None:
            mask = build_mask_from_spec(PandasMaskEngine, data, mask_spec)
            data = data.copy()
            data[source_col] = data[source_col].where(mask)

        # No partition columns: fold the whole table into one group via a constant helper column.
        helper_col: str | None = None
        if not partition_by:
            helper_col = unique_helper_name("__mloda_global_partition__", set(data.columns) | {feature_name})
            if mask_spec is None:
                data = data.copy()
            if len(data) == 0:
                # A zero-row groupby yields no groups; seed one all-null row. Reindex keeps
                # numeric columns numeric (NaN), since an object column makes median warn.
                data = data.reindex(range(1))
            data[helper_col] = 0
            partition_by = [helper_col]

        if agg_type == "mode":
            result = cls._compute_mode(data, feature_name, source_col, partition_by)
        else:
            grouped = null_safe_groupby(data, partition_by, source_col)
            result = apply_null_safe_agg(grouped, PANDAS_AGG_FUNCS[agg_type], agg_type).reset_index()
            result = result.rename(columns={source_col: feature_name})

            coerce_count_dtype(result, feature_name, agg_type)

        if helper_col is not None:
            result = result.drop(columns=[helper_col])

        return result

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
            unique_parts = data[partition_by].drop_duplicates().reset_index(drop=True).copy()
            unique_parts[feature_name] = unique_parts[source_col].where(unique_parts[source_col].notna(), pd.NA)
            return unique_parts

        winners = compute_mode_winners(data, source_col, partition_by)
        winners = winners.rename(columns={source_col: feature_name})

        all_partitions = data[partition_by].drop_duplicates().copy()
        all_partitions[feature_name] = pd.NA

        combined = pd.concat([winners, all_partitions], ignore_index=True, sort=False)
        return combined.groupby(partition_by, dropna=False, as_index=False)[feature_name].first().reset_index(drop=True)
