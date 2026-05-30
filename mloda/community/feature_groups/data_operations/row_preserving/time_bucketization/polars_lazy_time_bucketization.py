"""Polars Lazy implementation of time bucketization.

Uses Polars' native ``dt.truncate`` (floor) and ``dt.round`` (half-up by
default in Polars). Polars has no built-in ``ceil_temporal``; the
workaround stays inside this file: ``ceiled = when(col == floored) then
floored otherwise floored.offset_by(duration)`` for fixed-freq units, and
``floored + 1 bucket`` (always) for calendar units (matching PyArrow's
``ceil_temporal(ceil_is_strictly_greater=False)`` quirk).
"""

from __future__ import annotations

import polars as pl

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.base import (
    TIME_BUCKETIZATION_OPS,
    TimeBucketizationFeatureGroup,
)

# Polars duration aliases for each unit. Polars' ``dt.truncate('1w')`` is
# Monday-anchored, which matches the ISO week convention pinned by the FG.
_POLARS_UNIT_ALIASES: dict[str, str] = {
    "minute": "m",
    "hour": "h",
    "day": "d",
    "week": "w",
    "month": "mo",
    "year": "y",
}

# Calendar units whose ``ceil`` always advances by one bucket even on
# aligned input (matches PyArrow's ``ceil_temporal`` behaviour for
# ``week`` / ``month`` / ``year``). Fixed-freq units are idempotent on
# aligned input.
_CALENDAR_CEIL_ALWAYS_ADVANCES: frozenset[str] = frozenset({"week", "month", "year"})


def _duration_token(n: int, unit: str) -> str:
    """Format the Polars duration token for ``(n, unit)`` (e.g. ``5m``, ``1d``)."""
    return f"{n}{_POLARS_UNIT_ALIASES[unit]}"


class PolarsLazyTimeBucketization(TimeBucketizationFeatureGroup):
    """Polars-lazy backend for time bucketization."""

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PolarsLazyDataFrame}

    @classmethod
    def _assert_source_column_is_timestamp(cls, data: pl.LazyFrame, source_col: str) -> None:
        schema = data.collect_schema()
        if source_col not in schema:
            raise ValueError(
                f"Source column {source_col!r} is not present in the Polars LazyFrame; available: {list(schema)}."
            )
        dtype = schema[source_col]
        if not (dtype == pl.Datetime or (hasattr(dtype, "base_type") and dtype.base_type() == pl.Datetime)):
            cls._raise_non_timestamp_source(source_col, dtype)

    @classmethod
    def _compute_bucket(
        cls,
        data: pl.LazyFrame,
        feature_name: str,
        source_col: str,
        op: str,
        n: int,
        unit: str,
    ) -> pl.LazyFrame:
        col = pl.col(source_col)
        duration = _duration_token(n, unit)

        if op == "floor":
            expr = col.dt.truncate(duration)
        elif op == "ceil":
            expr = cls._ceil_expr(col, duration, unit)
        elif op == "round":
            # Polars ``dt.round`` is half-up by default for time series,
            # matching PyArrow's ``round_temporal``. Verified empirically:
            # 14:25 -> 14:30, 14:35 -> 14:40, 14:45 -> 14:50.
            expr = col.dt.round(duration)
        else:
            raise ValueError(f"Unsupported bucket op {op!r} for Polars; supported: {sorted(TIME_BUCKETIZATION_OPS)}.")

        return data.with_columns(expr.alias(feature_name))

    @classmethod
    def _ceil_expr(cls, col: pl.Expr, duration: str, unit: str) -> pl.Expr:
        """Polars ceil expression matching PyArrow's quirky calendar-unit behaviour."""
        floored = col.dt.truncate(duration)
        if unit in _CALENDAR_CEIL_ALWAYS_ADVANCES:
            # Calendar units always advance even on aligned input.
            return floored.dt.offset_by(duration)
        # Fixed-freq units: idempotent on aligned input.
        return pl.when(col == floored).then(floored).otherwise(floored.dt.offset_by(duration))
