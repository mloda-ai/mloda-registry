"""Polars Lazy implementation for aggregation feature groups."""

from __future__ import annotations

from typing import Any

import polars as pl

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.aggregation.base import (
    AggregationFeatureGroup,
)
from mloda.community.feature_groups.data_operations.errors import unsupported_agg_type_error
from mloda.community.feature_groups.data_operations.mask_utils import _POLARS_MASK_TMP, apply_polars_mask

# Mapping from aggregation type to a Polars expression builder.
_POLARS_AGG_EXPRS: dict[str, Any] = {
    "sum": lambda col: pl.col(col).sum(),
    "avg": lambda col: pl.col(col).mean(),
    "mean": lambda col: pl.col(col).mean(),
    "count": lambda col: pl.col(col).count(),
    "min": lambda col: pl.col(col).min(),
    "max": lambda col: pl.col(col).max(),
    "std": lambda col: pl.col(col).std(ddof=0),
    "var": lambda col: pl.col(col).var(ddof=0),
    "std_pop": lambda col: pl.col(col).std(ddof=0),
    "std_samp": lambda col: pl.col(col).std(ddof=1),
    "var_pop": lambda col: pl.col(col).var(ddof=0),
    "var_samp": lambda col: pl.col(col).var(ddof=1),
    "median": lambda col: pl.col(col).median(),
    "nunique": lambda col: pl.col(col).drop_nulls().n_unique(),
    "first": lambda col: pl.col(col).drop_nulls().first(),
    "last": lambda col: pl.col(col).drop_nulls().last(),
}

_SUPPORTED_AGG_TYPES = {*_POLARS_AGG_EXPRS.keys(), "mode"}


def _mode_with_insertion_order(s: pl.Series) -> Any:
    """Return the mode of *s*, breaking ties by first-occurrence order (matching PyArrow)."""
    s_clean = s.drop_nulls()
    if len(s_clean) == 0:
        return None
    df = s_clean.to_frame("v").with_row_index("_order")
    counts = df.group_by("v").agg(
        pl.col("_order").min().alias("first_idx"),
        pl.len().alias("cnt"),
    )
    winner = counts.sort(["cnt", "first_idx"], descending=[True, False]).row(0)
    return winner[0]


class PolarsLazyAggregation(AggregationFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PolarsLazyDataFrame}

    @classmethod
    def _compute_group(
        cls,
        data: pl.LazyFrame,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
        mask_spec: list[tuple[str, str, Any]] | None = None,
    ) -> pl.LazyFrame:
        """Compute a group aggregation using Polars group_by().agg() (fully lazy)."""
        actual_source = source_col
        if mask_spec is not None:
            data, actual_source = apply_polars_mask(data, source_col, mask_spec)

        if agg_type == "mode":
            # Use value_counts with insertion-order tie-breaking to match PyArrow.
            # Within each group, assign a row number to track insertion order,
            # then pick the value with the highest count (and lowest row number on ties).
            expr = (
                pl.col(actual_source)
                .map_batches(
                    lambda s: _mode_with_insertion_order(s),
                    returns_scalar=True,
                )
                .alias(feature_name)
            )
        elif agg_type in _POLARS_AGG_EXPRS:
            raw_expr = _POLARS_AGG_EXPRS[agg_type](actual_source).alias(feature_name)
            if agg_type == "sum":
                # Polars sum() returns 0 for all-null groups; correct to null.
                has_values = pl.col(actual_source).count() > 0
                expr = (
                    pl.when(has_values)
                    .then(_POLARS_AGG_EXPRS[agg_type](actual_source))
                    .otherwise(None)
                    .alias(feature_name)
                )
            else:
                expr = raw_expr
        else:
            raise unsupported_agg_type_error(agg_type, _SUPPORTED_AGG_TYPES, framework="Polars")

        result = data.group_by(partition_by, maintain_order=True).agg(expr)
        if mask_spec is not None:
            result = result.drop(_POLARS_MASK_TMP, strict=False)
        return result
