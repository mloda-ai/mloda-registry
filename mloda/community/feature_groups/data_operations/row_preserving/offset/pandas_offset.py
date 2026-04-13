"""Pandas implementation for offset feature groups."""

from __future__ import annotations

import pandas as pd

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.helper_columns import unique_helper_name
from mloda.community.feature_groups.data_operations.row_preserving.offset.base import (
    OffsetFeatureGroup,
)
from mloda.community.feature_groups.data_operations.pandas_helpers import null_safe_groupby


class PandasOffset(OffsetFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame}

    @classmethod
    def _compute_offset(
        cls,
        data: pd.DataFrame,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        order_by: str,
        offset_type: str,
    ) -> pd.DataFrame:
        """Compute offset using pandas groupby().shift()."""
        data = data.copy()

        # Sort by partition + order_by (nulls last) to ensure correct offset
        null_sort = data[order_by].isna().astype(int)
        null_sort_col = unique_helper_name("__mloda_null_sort", data.columns)
        data[null_sort_col] = null_sort
        data = data.sort_values(partition_by + [null_sort_col, order_by])

        grouped = null_safe_groupby(data, partition_by, source_col)

        if offset_type.startswith("lag_"):
            offset_n = int(offset_type[len("lag_") :])
            data[feature_name] = grouped.shift(offset_n)
        elif offset_type.startswith("lead_"):
            offset_n = int(offset_type[len("lead_") :])
            data[feature_name] = grouped.shift(-offset_n)
        elif offset_type.startswith("diff_"):
            offset_n = int(offset_type[len("diff_") :])
            data[feature_name] = data[source_col] - grouped.shift(offset_n)
        elif offset_type.startswith("pct_change_"):
            offset_n = int(offset_type[len("pct_change_") :])
            prev = grouped.shift(offset_n)
            data[feature_name] = (data[source_col] - prev) / prev.replace(0, float("nan"))
        elif offset_type == "first_value":
            data[feature_name] = grouped.transform(lambda x: x.dropna().iloc[0] if x.dropna().any() else None)
        elif offset_type == "last_value":
            data[feature_name] = grouped.transform(lambda x: x.dropna().iloc[-1] if x.dropna().any() else None)
        else:
            raise ValueError(f"Unsupported offset type: {offset_type}")

        data = data.drop(columns=[null_sort_col])
        # Restore original row order
        data = data.sort_index()
        return data
