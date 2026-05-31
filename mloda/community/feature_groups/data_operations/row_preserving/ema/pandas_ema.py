"""Pandas implementation of EMA-by-time."""

from __future__ import annotations

import pandas as pd

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.ema.base import EmaFeatureGroup

_RN_COL = "__mloda_rn__"


class PandasEma(EmaFeatureGroup):
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
    def _compute_ema(
        cls,
        data: pd.DataFrame,
        feature_name: str,
        source_col: str,
        span: int,
        partition_by: list[str],
        order_by: str,
    ) -> pd.DataFrame:
        data = data.copy()

        # Tag original row order so we can restore it after sorting.
        data[_RN_COL] = range(len(data))

        ordered = data.sort_values(by=[*partition_by, order_by], na_position="last")

        def _ema(series: pd.Series) -> pd.Series:
            # adjust=False, nulls skipped in the recurrence; null input -> null output.
            return series.ewm(span=span, adjust=False, ignore_na=True).mean().mask(series.isna())

        if partition_by:
            ordered[feature_name] = ordered.groupby(partition_by, dropna=False)[source_col].transform(_ema)
        else:
            ordered[feature_name] = _ema(ordered[source_col])

        # Restore original row order, drop the helper column.
        result = ordered.sort_values(by=_RN_COL).drop(columns=[_RN_COL])
        return result
