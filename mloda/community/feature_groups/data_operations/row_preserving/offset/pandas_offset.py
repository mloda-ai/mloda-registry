"""Pandas implementation for offset feature groups."""

from __future__ import annotations

from typing import Any

import pandas as pd
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.helper_columns import unique_helper_name
from mloda.community.feature_groups.data_operations.pandas_helpers import null_safe_groupby
from mloda.community.feature_groups.data_operations.row_preserving.offset.base import (
    OffsetFeatureGroup,
)


def _widen(series: pd.Series) -> pd.Series:
    """Widen integers to 64 bits so subtraction cannot wrap; reject uint64 values of 2**63 or more."""
    dtype = series.dtype
    if not pd.api.types.is_integer_dtype(dtype):
        return series
    extension = isinstance(dtype, pd.api.extensions.ExtensionDtype)
    if dtype.itemsize >= 8 and dtype.kind == "u":
        if (series >= 2**63).any():
            raise ValueError(
                "Offset diff and pct_change on pandas cannot subtract a uint64 source value of 2**63 or more "
                "without wrapping: it does not fit int64 and pandas has no 128-bit integer; use Polars or DuckDB"
            )
        return series.astype("Int64")
    if dtype.itemsize >= 8:
        return series
    return series.astype("Int64" if extension else "int64")


def _nullable(series: pd.Series) -> pd.Series:
    """Map a numpy integer series to its nullable extension dtype so shifts keep the integer type."""
    dtype = series.dtype
    if isinstance(dtype, pd.api.extensions.ExtensionDtype) or not pd.api.types.is_integer_dtype(dtype):
        return series
    return series.astype(("U" if dtype.kind == "u" else "") + "Int" + str(dtype.itemsize * 8))


class PandasOffset(Implementation[OffsetFeatureGroup]):
    feature_group = OffsetFeatureGroup
    compute_framework = PandasDataFrame

    @classmethod
    def calculate(
        cls, group: type[OffsetFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.offset(data, features, compute_offset=cls._compute_offset)

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
        data = data.copy()

        null_sort_col = unique_helper_name("__mloda_null_sort", set(data.columns) | {feature_name})

        # Sort by partition + order_by (nulls last) to ensure correct offset
        null_sort = data[order_by].isna().astype(int)
        data[null_sort_col] = null_sort
        data = data.sort_values(partition_by + [null_sort_col, order_by])

        shift_col = unique_helper_name("__mloda_shift_src", set(data.columns) | {feature_name})
        unsigned_64 = (
            pd.api.types.is_unsigned_integer_dtype(data[source_col].dtype) and data[source_col].dtype.itemsize >= 8
        )
        if offset_type.startswith(("lag_", "lead_", "diff_")) or unsigned_64:
            data[shift_col] = _nullable(data[source_col])
        else:
            data[shift_col] = data[source_col]
        grouped = null_safe_groupby(data, partition_by, shift_col)

        if offset_type.startswith("lag_"):
            offset_n = int(offset_type[len("lag_") :])
            data[feature_name] = grouped.shift(offset_n)
        elif offset_type.startswith("lead_"):
            offset_n = int(offset_type[len("lead_") :])
            data[feature_name] = grouped.shift(-offset_n)
        elif offset_type.startswith("diff_"):
            offset_n = int(offset_type[len("diff_") :])
            data[feature_name] = _widen(data[source_col]) - _widen(grouped.shift(offset_n))
        elif offset_type.startswith("pct_change_"):
            offset_n = int(offset_type[len("pct_change_") :])
            prev = _widen(grouped.shift(offset_n))
            data[feature_name] = (_widen(data[source_col]) - prev) / prev.replace(0, float("nan"))
        elif offset_type == "first_value":
            data[feature_name] = null_safe_groupby(data, partition_by, source_col).transform("first")
        elif offset_type == "last_value":
            data[feature_name] = null_safe_groupby(data, partition_by, source_col).transform("last")
        else:
            raise ValueError(f"Unsupported offset type: {offset_type}")

        data = data.drop(columns=[null_sort_col, shift_col])
        # Restore original row order
        data = data.sort_index()
        return data
