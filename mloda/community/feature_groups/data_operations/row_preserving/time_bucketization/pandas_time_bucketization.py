"""Pandas implementation of time bucketization.

Fixed-freq units (``minute`` / ``hour`` / ``day``) delegate to pandas'
``Series.dt.floor`` / ``ceil`` / ``round``. Calendar units (``week``,
``month``, ``year``) use ``PeriodIndex`` floor and a one-bucket
``DateOffset`` for ceil, plus a half-up midpoint comparison for round.

To match PyArrow's quirky calendar-unit ceil (which advances even on
aligned input for ``week`` / ``month`` / ``year``, but is idempotent for
``day``), the ceil path branches on unit family.
"""

from __future__ import annotations

import pandas as pd

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.base import (
    TIME_BUCKETIZATION_OPS,
    TimeBucketizationFeatureGroup,
)

# Pandas frequency aliases for fixed-freq dt floor/ceil/round.
_FIXED_FREQ_ALIASES: dict[str, str] = {
    "minute": "min",
    "hour": "h",
    "day": "D",
}

# Pandas period-based floor for calendar units (n=1 only). For ISO weeks
# (Monday start) we use ``W-SUN`` (week ending Sunday), so a Sunday rolls
# into the *previous* week's Monday and a Monday is its own start. The
# ``W-MON`` alias would instead mean weeks ending Monday, which floors a
# Sunday to the following Tuesday's preceding Monday — not what ISO wants.
_PERIOD_FREQ: dict[str, str] = {
    "week": "W-SUN",
    "month": "M",
    "year": "Y",
}

# Calendar units whose ceil always advances by one bucket even on aligned
# input (matching PyArrow's ``ceil_temporal`` behaviour).
_CALENDAR_CEIL_ALWAYS_ADVANCES: frozenset[str] = frozenset({"week", "month", "year"})


def _calendar_offset(unit: str) -> pd.DateOffset | pd.Timedelta:
    """One-bucket DateOffset / Timedelta for a calendar unit."""
    if unit == "week":
        return pd.Timedelta(days=7)
    if unit == "month":
        return pd.DateOffset(months=1)
    if unit == "year":
        return pd.DateOffset(years=1)
    raise ValueError(f"Calendar offset requested for non-calendar unit: {unit!r}")


def _calendar_floor(series: pd.Series, unit: str) -> pd.Series:
    """Floor ``series`` to the start of its containing calendar bucket.

    Preserves the input tz: pandas' ``Period.to_timestamp`` drops tz, so the
    helper re-localises to the original.
    """
    freq = _PERIOD_FREQ[unit]
    tz = series.dt.tz
    naive = series.dt.tz_localize(None) if tz is not None else series
    floored_naive = naive.dt.to_period(freq).dt.to_timestamp()
    if tz is not None:
        return floored_naive.dt.tz_localize(tz)
    return floored_naive


class PandasTimeBucketization(TimeBucketizationFeatureGroup):
    """Pandas backend for time bucketization."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame}

    @classmethod
    def _assert_source_column_is_timestamp(cls, data: pd.DataFrame, source_col: str) -> None:
        series = data[source_col]
        if not pd.api.types.is_datetime64_any_dtype(series):
            cls._raise_non_timestamp_source(source_col, series.dtype)

    @classmethod
    def _compute_bucket(
        cls,
        data: pd.DataFrame,
        feature_name: str,
        source_col: str,
        op: str,
        n: int,
        unit: str,
    ) -> pd.DataFrame:
        data = data.copy()
        col = data[source_col]

        if op == "floor":
            result = cls._floor_series(col, n, unit)
        elif op == "ceil":
            result = cls._ceil_series(col, n, unit)
        elif op == "round":
            result = cls._round_series(col, n, unit)
        else:
            raise ValueError(f"Unsupported bucket op {op!r} for Pandas; supported: {sorted(TIME_BUCKETIZATION_OPS)}.")

        data[feature_name] = result
        return data

    # -- Op implementations --------------------------------------------------

    @classmethod
    def _floor_series(cls, col: pd.Series, n: int, unit: str) -> pd.Series:
        if unit in _FIXED_FREQ_ALIASES:
            return col.dt.floor(f"{n}{_FIXED_FREQ_ALIASES[unit]}")
        return _calendar_floor(col, unit)

    @classmethod
    def _ceil_series(cls, col: pd.Series, n: int, unit: str) -> pd.Series:
        if unit in _FIXED_FREQ_ALIASES:
            return col.dt.ceil(f"{n}{_FIXED_FREQ_ALIASES[unit]}")
        # Calendar units: always advance one bucket (matches PyArrow).
        floored = _calendar_floor(col, unit)
        offset = _calendar_offset(unit)
        return floored + offset

    @classmethod
    def _round_series(cls, col: pd.Series, n: int, unit: str) -> pd.Series:
        """Round half-UP (every midpoint rolls to the higher bucket).

        Pandas' ``Series.dt.round`` defaults to half-to-even (banker's), but
        the cross-framework spec is half-up (matches PyArrow's
        ``round_temporal`` and Polars' ``dt.round``). For fixed-freq units
        we still rely on ``dt.floor`` / ``dt.ceil`` to get the bracket
        bucket and then pick via midpoint comparison.
        """
        if unit in _FIXED_FREQ_ALIASES:
            floored = col.dt.floor(f"{n}{_FIXED_FREQ_ALIASES[unit]}")
            ceiled = col.dt.ceil(f"{n}{_FIXED_FREQ_ALIASES[unit]}")
        else:
            floored = _calendar_floor(col, unit)
            ceiled = floored + _calendar_offset(unit)

        # Half-up: choose ceiled when the input is at or past the midpoint.
        diff_to_floor = col - floored
        diff_floor_to_ceil = ceiled - floored
        is_at_or_past_midpoint = diff_to_floor * 2 >= diff_floor_to_ceil
        rounded = floored.where(~is_at_or_past_midpoint, ceiled)
        return rounded.mask(col.isna())
