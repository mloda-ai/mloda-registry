"""Pandas implementation for rank feature groups."""

from __future__ import annotations

from typing import Any

import pandas as pd
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.pandas_helpers import (
    null_safe_groupby,
    nulls_last_rank_key,
)
from mloda.community.feature_groups.data_operations.row_preserving.rank.base import (
    RankFeatureGroup,
)

_PANDAS_RANK_METHODS: dict[str, str] = {
    "row_number": "first",
    "rank": "min",
    "dense_rank": "dense",
}


class PandasRank(Implementation[RankFeatureGroup]):
    feature_group = RankFeatureGroup
    compute_framework = PandasDataFrame
    divergent = {
        "rank": "float64 cannot tell NaN from null, so they tie",
        "dense_rank": "float64 cannot tell NaN from null, so they tie",
        "percent_rank": "float64 cannot tell NaN from null, so they tie",
    }

    @classmethod
    def calculate(
        cls, group: type[RankFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.rank(data, features, compute_rank=cls._compute_rank)

    @classmethod
    def _compute_rank(
        cls,
        data: pd.DataFrame,
        feature_name: str,
        partition_by: list[str],
        order_by: str,
        rank_type: str,
    ) -> pd.DataFrame:
        data = data.copy()
        original = data
        if pd.api.types.is_extension_array_dtype(data[order_by]):
            data = data.copy()
            data[order_by] = nulls_last_rank_key(data[order_by], ascending=not rank_type.startswith("top_"))

        if rank_type in _PANDAS_RANK_METHODS:
            method = _PANDAS_RANK_METHODS[rank_type]
            data[feature_name] = (
                null_safe_groupby(data, partition_by, order_by)
                .rank(method=method, ascending=True, na_option="bottom")
                .astype("int64")
            )
        elif rank_type == "percent_rank":
            # percent_rank = (rank - 1) / (count - 1)
            rank_col = null_safe_groupby(data, partition_by, order_by).rank(
                method="min", ascending=True, na_option="bottom"
            )
            group_size = null_safe_groupby(data, partition_by, order_by).transform("size")
            data[feature_name] = ((rank_col - 1) / (group_size - 1)).fillna(0.0)
        elif rank_type.startswith("ntile_"):
            ntile_n = int(rank_type[len("ntile_") :])
            rank_col = null_safe_groupby(data, partition_by, order_by).rank(
                method="first", ascending=True, na_option="bottom"
            )
            group_size = null_safe_groupby(data, partition_by, order_by).transform("size")
            data[feature_name] = (((rank_col - 1) * ntile_n) // group_size + 1).astype("int64")
        elif rank_type.startswith("top_"):
            top_n = int(rank_type[len("top_") :])
            row_num = null_safe_groupby(data, partition_by, order_by).rank(
                method="first", ascending=False, na_option="bottom"
            )
            data[feature_name] = row_num <= top_n
        elif rank_type.startswith("bottom_"):
            bottom_n = int(rank_type[len("bottom_") :])
            row_num = null_safe_groupby(data, partition_by, order_by).rank(
                method="first", ascending=True, na_option="bottom"
            )
            data[feature_name] = row_num <= bottom_n
        else:
            raise ValueError(f"Unsupported rank type: {rank_type}")

        if original is not data:
            data[order_by] = original[order_by]
        return data
