"""Pandas implementation of resample.

Floors the time column with ``Series.dt.floor`` (same fixed-freq aliases as
``pandas_time_bucketization``), groups by ``(*partition_by, bucket)`` with
``dropna=False`` and aggregates via the shared pandas helpers. ``sum`` uses
``min_count=1`` so an all-null bucket yields NaN (then coerced to ``None``),
matching the PyArrow oracle rather than pandas' default ``0.0``.
"""

from __future__ import annotations

import pandas as pd

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.pandas_helpers import (
    PANDAS_AGG_FUNCS,
    apply_null_safe_agg,
    coerce_count_dtype,
    null_safe_groupby,
)
from mloda.community.feature_groups.data_operations.row_changing.resample.base import (
    RESAMPLE_AGGS,
    ResampleFeatureGroup,
)
from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.pandas_time_bucketization import (
    _FIXED_FREQ_ALIASES,
)


class PandasResample(ResampleFeatureGroup):
    """Pandas backend for resample."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame}

    @classmethod
    def _assert_time_column_present(cls, data: pd.DataFrame, time_column: str) -> None:
        if time_column not in data.columns:
            raise ValueError(
                f"time_column {time_column!r} is not present in the Pandas DataFrame; available: {list(data.columns)}."
            )

    @classmethod
    def _assert_source_column_present(cls, data: pd.DataFrame, source_col: str) -> None:
        if source_col not in data.columns:
            raise ValueError(
                f"Source column {source_col!r} is not present in the Pandas DataFrame; available: {list(data.columns)}."
            )

    @classmethod
    def _compute_resample(
        cls,
        data: pd.DataFrame,
        feature_name: str,
        source_col: str,
        time_column: str,
        partition_by: list[str],
        n: int,
        unit: str,
        agg: str,
    ) -> pd.DataFrame:
        pandas_func = PANDAS_AGG_FUNCS.get(agg)
        if pandas_func is None:
            raise ValueError(f"Unsupported resample agg {agg!r} for Pandas; supported: {sorted(RESAMPLE_AGGS)}.")

        data = data.copy()
        # Floor the time column in place (bucket start keeps the original name).
        data[time_column] = data[time_column].dt.floor(f"{n}{_FIXED_FREQ_ALIASES[unit]}")

        keys = [*partition_by, time_column]
        grouped = null_safe_groupby(data, keys, source_col)
        result = apply_null_safe_agg(grouped, pandas_func, agg).reset_index()
        result = result.rename(columns={source_col: feature_name})

        coerce_count_dtype(result, feature_name, agg)

        if agg != "count":
            # NaN (all-null sum / mean) must surface as real None to match the
            # PyArrow oracle (the harness asserts identity ``is None``).
            result[feature_name] = result[feature_name].astype(object).where(result[feature_name].notna(), None)

        return result
