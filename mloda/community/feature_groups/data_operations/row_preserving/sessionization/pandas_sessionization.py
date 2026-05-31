"""Pandas implementation of gap-threshold sessionization."""

from __future__ import annotations

import pandas as pd

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.sessionization.base import (
    SessionizationFeatureGroup,
)

_RN_COL = "__mloda_rn__"


class PandasSessionization(SessionizationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame}

    @classmethod
    def _assert_source_column_present(cls, data: pd.DataFrame, order_col: str) -> None:
        if order_col not in data.columns:
            raise ValueError(
                f"Source column {order_col!r} is not present in the pandas DataFrame; available: {list(data.columns)}."
            )

    @classmethod
    def _compute_session(
        cls,
        data: pd.DataFrame,
        feature_name: str,
        order_col: str,
        threshold_seconds: int,
        partition_by: list[str],
    ) -> pd.DataFrame:
        data = data.copy()

        # Tag original row order so we can restore it after sorting.
        data[_RN_COL] = range(len(data))

        ordered = data.sort_values(by=[*partition_by, order_col], na_position="last")

        if partition_by:
            gap = ordered.groupby(partition_by, dropna=False)[order_col].diff()
        else:
            gap = ordered[order_col].diff()

        threshold = pd.Timedelta(seconds=threshold_seconds)
        is_new = gap.isna() | (gap > threshold)
        ordered[feature_name] = is_new.cumsum().astype("int64") - 1

        # Restore original row order, drop the helper column.
        result = ordered.sort_values(by=_RN_COL).drop(columns=[_RN_COL])
        return result
