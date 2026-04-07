"""Pandas implementation for rank feature groups."""

from __future__ import annotations

import pandas as pd

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.rank.base import (
    RankFeatureGroup,
)
from mloda.community.feature_groups.data_operations.pandas_helpers import null_safe_groupby

_PANDAS_RANK_METHODS: dict[str, str] = {
    "row_number": "first",
    "rank": "min",
    "dense_rank": "dense",
}


class PandasRank(RankFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame}

    @classmethod
    def _compute_rank(
        cls,
        data: pd.DataFrame,
        feature_name: str,
        partition_by: list[str],
        order_by: str,
        rank_type: str,
    ) -> pd.DataFrame:
        """Compute rank using pandas groupby().rank()."""
        data = data.copy()

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

        return data
