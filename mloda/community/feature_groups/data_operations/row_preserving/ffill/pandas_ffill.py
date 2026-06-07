"""Pandas implementation of ffill-by-time."""

from __future__ import annotations

import pandas as pd

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.helper_columns import unique_helper_name
from mloda.community.feature_groups.data_operations.pandas_helpers import assert_source_col_present
from mloda.community.feature_groups.data_operations.row_preserving.ffill.base import FfillFeatureGroup


class PandasFfill(FfillFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame}

    @classmethod
    def _assert_source_column_present(cls, data: pd.DataFrame, source_col: str) -> None:
        assert_source_col_present(data, source_col)

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

        rn_col = unique_helper_name("__mloda_rn__", set(data.columns) | {feature_name})

        # Tag original row order so we can restore it after sorting.
        data[rn_col] = range(len(data))

        ordered = data.sort_values(by=[*partition_by, order_by], na_position="last")

        if partition_by:
            filled = ordered.groupby(partition_by, dropna=False)[source_col].ffill()
        else:
            filled = ordered[source_col].ffill()

        ordered[feature_name] = filled

        # Restore original row order, drop the helper column.
        result = ordered.sort_values(by=rn_col).drop(columns=[rn_col])
        return result
