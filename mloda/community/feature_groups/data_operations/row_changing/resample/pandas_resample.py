"""Pandas implementation of resample.

Floors the time column with ``Series.dt.floor`` (the same shared fixed-freq
aliases used for bucketization), groups by ``(*partition_by, bucket)`` with
``dropna=False`` and aggregates via the shared pandas helpers. ``sum`` uses
``min_count=1`` so an all-null bucket yields NaN (then coerced to ``None``),
matching the PyArrow oracle rather than pandas' default ``0.0``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.pandas_helpers import (
    FIXED_FREQ_ALIASES,
    PANDAS_AGG_FUNCS,
    apply_null_safe_agg,
    coerce_count_dtype,
    null_safe_groupby,
)
from mloda.community.feature_groups.data_operations.row_changing.resample.base import (
    RESAMPLE_AGGS,
    ResampleAgg,
    ResampleFeatureGroup,
)


class PandasResample(Implementation[ResampleFeatureGroup]):
    """Pandas backend for resample."""

    feature_group = ResampleFeatureGroup
    compute_framework = PandasDataFrame
    divergent = {
        "sum": "skips NaN; null for an all-NaN bucket",
        "mean": "skips NaN; null for an all-NaN bucket",
        "count": "does not count NaN",
        "min": "skips NaN; null for an all-NaN bucket",
        "max": "skips NaN; null for an all-NaN bucket",
    }

    @classmethod
    def calculate(
        cls, group: type[ResampleFeatureGroup], data: Any, features: FeatureSet, framework: ComputeFramework | None
    ) -> Any:
        return group.resample(data, features, compute_resample=cls._compute_resample)

    @classmethod
    def _compute_resample(
        cls,
        data: pd.DataFrame,
        aggs: list[ResampleAgg],
        time_column: str,
        partition_by: list[str],
        n: int,
        unit: str,
    ) -> pd.DataFrame:
        for spec in aggs:
            if spec.agg not in PANDAS_AGG_FUNCS:
                raise ValueError(
                    f"Unsupported resample agg {spec.agg!r} for Pandas; supported: {sorted(RESAMPLE_AGGS)}."
                )

        data = data.copy()
        # Floor the time column in place (bucket start keeps the original name).
        data[time_column] = data[time_column].dt.floor(f"{n}{FIXED_FREQ_ALIASES[unit]}")

        keys = [*partition_by, time_column]
        result: pd.DataFrame | None = None
        for spec in aggs:
            grouped = null_safe_groupby(data, keys, spec.source_col)
            part = apply_null_safe_agg(grouped, PANDAS_AGG_FUNCS[spec.agg], spec.agg).reset_index()
            part = part.rename(columns={spec.source_col: spec.feature_name})
            coerce_count_dtype(part, spec.feature_name, spec.agg)

            if spec.agg != "count":
                # NaN (all-null sum / mean) must surface as real None to match the
                # PyArrow oracle (the harness asserts identity ``is None``).
                part[spec.feature_name] = (
                    part[spec.feature_name].astype(object).where(part[spec.feature_name].notna(), None)
                )

            if result is None:
                result = part
            else:
                # Same data and keys, so the group order is identical: attach positionally (keys may be null).
                result[spec.feature_name] = part[spec.feature_name].to_numpy()

        assert result is not None
        return result
