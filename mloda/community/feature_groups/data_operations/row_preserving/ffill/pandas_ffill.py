"""Pandas implementation of ffill-by-time."""

from __future__ import annotations

import pandas as pd

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.ffill.base import FfillFeatureGroup

_RN_COL = "__mloda_rn__"


class PandasFfill(FfillFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame}

    @classmethod
    def _assert_source_column_present(cls, data: pd.DataFrame, source_col: str) -> None:
        if source_col not in data.columns:
            raise ValueError(
                f"Source column {source_col!r} is not present in the pandas DataFrame; available: {list(data.columns)}."
            )

    @classmethod
    def _compute_ffill(
        cls,
        data: pd.DataFrame,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        order_by: str,
    ) -> pd.DataFrame:
        data = data.copy()

        # Tag original row order so we can restore it after sorting.
        data[_RN_COL] = range(len(data))

        ordered = data.sort_values(by=[*partition_by, order_by], na_position="last")

        if partition_by:
            filled = ordered.groupby(partition_by, dropna=False)[source_col].ffill()
        else:
            filled = ordered[source_col].ffill()

        ordered[feature_name] = filled

        # Restore original row order, drop the helper column.
        result = ordered.sort_values(by=_RN_COL).drop(columns=[_RN_COL])
        return result
