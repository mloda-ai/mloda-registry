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
    # Pandas .rolling(window="...", on=ts) only accepts fixed-frequency units. Month/year
    # are calendar-anchored and would require a per-row Python loop, which defeats the
    # point of running inside pandas. They are rejected at match time. See
    # known-divergences.md.
    SUPPORTED_TIME_UNITS: set[str] = {"second", "minute", "hour", "day", "week"}

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
        # Safe to mutate: data was already copied above (data = data.copy()).
        #
        # Collision case: when source_col == order_by and a mask is applied,
        # the reference treats masked rows as having null ``order_by`` (because
        # mask writes null into source_col, which is also order_by). Pandas'
        # native ``rolling(on=ts)`` cannot simulate this without a Python loop
        # (which would defeat the point of running inside pandas). For non-time
        # frames the order_by clobber would also break the sort; reject the
        # combo for time frames at runtime. See known-divergences.md.
        agg_col = source_col
        if mask_spec is not None:
            mask = build_mask_from_spec(PandasMaskEngine, data, mask_spec)
            if source_col == order_by:
                if frame_type == "time":
                    raise ValueError(
                        "Pandas frame aggregate (time frame): mask + source_col == order_by "
                        f"({source_col!r}) is unsupported. The reference semantic requires "
                        "treating masked rows as having null order_by, which pandas "
                        "rolling(on=...) cannot express natively. See known-divergences.md."
                    )
                # Non-time frames: aggregate the masked values in a temp column;
                # order_by is preserved because the sort already happened above.
                agg_col = "__mloda_masked_source__"
                data[agg_col] = data[source_col].where(mask)
            else:
                data[source_col] = data[source_col].where(mask)

        grouped = null_safe_groupby(data, partition_by, agg_col)

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
            if unit not in cls._FIXED_FREQ_CODES:
                # Defense-in-depth: month/year are rejected at match time via
                # SUPPORTED_TIME_UNITS. Hitting this path means a caller bypassed
                # match_feature_group_criteria.
                raise unsupported_frame_type_error(
                    f"time:{unit}",
                    {f"time:{u}" for u in cls._FIXED_FREQ_CODES},
                    framework="Pandas",
                )
            if data[order_by].isna().any():
                # pandas groupby().rolling(on=ts) raises "ts values must not have NaT".
                # Convert that cryptic error into an explicit refusal naming the column.
                # See known-divergences.md.
                raise ValueError(
                    f"Pandas frame aggregate (time frame): order_by column {order_by!r} "
                    "contains null/NaT values, which pandas groupby().rolling(on=...) does "
                    "not support. See known-divergences.md."
                )
            data[feature_name] = cls._compute_fixed_freq_time(
                data, agg_col, partition_by, order_by, agg_type, size, unit, min_periods
            )
            coerce_count_dtype(data, feature_name, agg_type)
            data = data.sort_values(by=rn_col)
            drop_cols = [rn_col]
            if mask_spec is not None and source_col == order_by:
                drop_cols.append(agg_col)
            data = data.drop(columns=drop_cols)
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

        # Restore original row order and drop helper columns
        data = data.sort_values(by=rn_col)
        drop_cols = [rn_col]
        if mask_spec is not None and source_col == order_by:
            drop_cols.append(agg_col)
        data = data.drop(columns=drop_cols)

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
