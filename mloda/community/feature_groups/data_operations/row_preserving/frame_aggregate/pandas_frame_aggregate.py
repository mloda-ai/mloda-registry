"""Pandas implementation for frame aggregate feature groups."""

from __future__ import annotations

from typing import Any

import pandas as pd

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.errors import (
    unsupported_agg_type_error,
    unsupported_frame_type_error,
)
from mloda.community.feature_groups.data_operations.mask_utils import build_mask_from_spec
from mloda.community.feature_groups.data_operations.reserved_columns import assert_no_reserved_columns
from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.base import (
    FrameAggregateFeatureGroup,
)
from mloda.community.feature_groups.data_operations.pandas_helpers import (
    PANDAS_AGG_FUNCS,
    coerce_count_dtype,
    null_safe_groupby,
)
from mloda_plugins.compute_framework.base_implementations.pandas.pandas_mask_engine import (
    PandasMaskEngine,
)

_PANDAS_FRAME_AGG_FUNCS: dict[str, str] = {
    **PANDAS_AGG_FUNCS,
    "std": "std",
    "var": "var",
    "median": "median",
}


class PandasFrameAggregate(FrameAggregateFeatureGroup):
    SUPPORTED_FRAME_TYPES = {"rolling", "time", "cumulative", "expanding"}

    # Pandas .rolling(window="...", on=ts) only accepts these fixed-frequency units.
    # Month/year are calendar-anchored and require a per-row Python loop instead.
    _FIXED_FREQ_CODES: dict[str, str] = {
        "second": "s",
        "minute": "min",
        "hour": "h",
        "day": "D",
        "week": "W",
    }

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame}

    @classmethod
    def _compute_frame(
        cls,
        data: pd.DataFrame,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        order_by: str,
        agg_type: str,
        frame_type: str,
        frame_size: int | None = None,
        frame_unit: str | None = None,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> pd.DataFrame:
        assert_no_reserved_columns(data.columns, framework="Pandas", operation="frame aggregate")

        pandas_func = _PANDAS_FRAME_AGG_FUNCS.get(agg_type)
        if pandas_func is None:
            raise unsupported_agg_type_error(
                agg_type,
                _PANDAS_FRAME_AGG_FUNCS.keys(),
                framework="Pandas",
                operation="frame aggregate",
            )

        data = data.copy()

        # Save original row order so we can restore it after sorting
        rn_col = "__mloda_rn__"
        data[rn_col] = range(len(data))

        data = data.sort_values(by=[*partition_by, order_by], na_position="last")

        # PyArrow parity: the reference applies masks before aggregation but
        # sorts on unmasked values. Apply mask AFTER sorting so that sort
        # order uses original (unmasked) values to match this behavior.
        # This matters when order_by == source_col.
        # Safe to mutate: data was already copied above (data = data.copy()).
        if mask_spec is not None:
            mask = build_mask_from_spec(PandasMaskEngine, data, mask_spec)
            data[source_col] = data[source_col].where(mask)

        grouped = null_safe_groupby(data, partition_by, source_col)

        # std/var require at least 2 observations for a meaningful result
        min_periods = 2 if agg_type in ("std", "var") else 1
        reset_levels = list(range(len(partition_by)))

        if frame_type in ("cumulative", "expanding"):
            window_obj = grouped.expanding(min_periods=min_periods)
        elif frame_type == "rolling":
            window = int(frame_size) if frame_size is not None else 1
            window_obj = grouped.rolling(window=window, min_periods=min_periods)
        elif frame_type == "time":
            size = int(frame_size) if frame_size is not None else 1
            unit = str(frame_unit or "day")
            if unit in cls._FIXED_FREQ_CODES:
                data[feature_name] = cls._compute_fixed_freq_time(
                    data, source_col, partition_by, order_by, agg_type, size, unit, min_periods
                )
            else:
                data[feature_name] = cls._compute_calendar_time(
                    data, source_col, partition_by, order_by, agg_type, size, unit
                )
            coerce_count_dtype(data, feature_name, agg_type)
            data = data.sort_values(by=rn_col)
            data = data.drop(columns=[rn_col])
            return data
        else:
            raise unsupported_frame_type_error(
                frame_type,
                cls.SUPPORTED_FRAME_TYPES,
                framework="Pandas",
            )

        if agg_type in ("std", "var"):
            result = getattr(window_obj, pandas_func)(ddof=0).reset_index(level=reset_levels, drop=True)
        else:
            result = getattr(window_obj, pandas_func)().reset_index(level=reset_levels, drop=True)

        data[feature_name] = result
        coerce_count_dtype(data, feature_name, agg_type)

        # Restore original row order and drop helper column
        data = data.sort_values(by=rn_col)
        data = data.drop(columns=[rn_col])

        return data

    @classmethod
    def _compute_fixed_freq_time(
        cls,
        data: pd.DataFrame,
        source_col: str,
        partition_by: list[str],
        order_by: str,
        agg_type: str,
        size: int,
        unit: str,
        min_periods: int,
    ) -> pd.Series:
        """Time window using pandas native rolling on a fixed-frequency offset.

        Uses ``closed="both"`` so the window includes the lower bound, matching
        the reference's ``ts_other >= ts - delta`` (inclusive) semantics.

        ``data`` is already sorted by ``[*partition_by, order_by]``. The
        ``groupby().rolling()`` output preserves that order positionally, so
        the result values are assigned by position (via ``.values``) rather
        than by index, because rolling replaces the index with ``order_by``.
        """
        pandas_func = _PANDAS_FRAME_AGG_FUNCS[agg_type]
        # Convert weeks to days (pandas treats "W" as an anchored, non-fixed offset).
        if unit == "week":
            window_str = f"{size * 7}D"
        else:
            window_str = f"{size}{cls._FIXED_FREQ_CODES[unit]}"
        by: str | list[str] = partition_by[0] if len(partition_by) == 1 else partition_by

        rolling_obj = data.groupby(by, dropna=False).rolling(
            window=window_str,
            on=order_by,
            closed="both",
            min_periods=min_periods,
        )[source_col]

        if agg_type in ("std", "var"):
            rolled = getattr(rolling_obj, pandas_func)(ddof=0)
        else:
            rolled = getattr(rolling_obj, pandas_func)()
        return pd.Series(rolled.values, index=data.index)

    @classmethod
    def _compute_calendar_time(
        cls,
        data: pd.DataFrame,
        source_col: str,
        partition_by: list[str],
        order_by: str,
        agg_type: str,
        size: int,
        unit: str,
    ) -> pd.Series:
        """Calendar-anchored time window for month/year units.

        Pandas .rolling does not accept non-fixed offsets; iterate per
        partition and compute each window via ``relativedelta`` so months
        and years respect calendar boundaries.
        """
        from dateutil.relativedelta import relativedelta

        from mloda.community.feature_groups.data_operations.aggregate_helpers import aggregate

        if unit == "month":
            delta = relativedelta(months=size)
        else:
            delta = relativedelta(years=size)

        by: str | list[str] = partition_by[0] if len(partition_by) == 1 else partition_by
        result = pd.Series(index=data.index, dtype=object)

        for _, group in data.groupby(by, dropna=False, sort=False):
            sorted_group = group.sort_values(
                by=order_by,
                na_position="last",
                kind="mergesort",
            )
            order_vals = sorted_group[order_by].tolist()
            source_vals = [None if pd.isna(v) else v for v in sorted_group[source_col].tolist()]
            indices = sorted_group.index.tolist()
            for pos, idx in enumerate(indices):
                current = order_vals[pos]
                if pd.isna(current):
                    window = [source_vals[pos]]
                else:
                    cutoff = current - delta
                    window = [
                        source_vals[i] for i in range(pos + 1) if not pd.isna(order_vals[i]) and order_vals[i] >= cutoff
                    ]
                result.at[idx] = aggregate(window, agg_type)
        return result
