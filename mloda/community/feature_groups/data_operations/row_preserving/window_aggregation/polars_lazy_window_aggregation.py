"""Polars Lazy implementation for window aggregation feature groups."""

from __future__ import annotations

from typing import Any, Set, Type, Union

import polars as pl
import pyarrow as pa

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.base import ColumnTypeCategory, register_type_checker
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
    WindowAggregationFeatureGroup,
)

# Register Polars type checkers with the shared registry.
register_type_checker(
    "polars",
    {
        ColumnTypeCategory.NUMERIC: lambda dtype: bool(dtype.is_numeric()),
        ColumnTypeCategory.STRING: lambda dtype: bool(dtype == pl.Utf8 or dtype == pl.String),
        ColumnTypeCategory.DATETIME: lambda dtype: bool(dtype == pl.Datetime or dtype == pl.Date),
        ColumnTypeCategory.ANY: lambda dtype: True,
    },
)

# Mapping from aggregation type to a Polars expression builder.
# Each callable takes a column name and returns a Polars Expr.
_POLARS_AGG_EXPRS: dict[str, Any] = {
    "sum": lambda col: pl.col(col).sum(),
    "avg": lambda col: pl.col(col).mean(),
    "count": lambda col: pl.col(col).count(),
    "min": lambda col: pl.col(col).min(),
    "max": lambda col: pl.col(col).max(),
}


class PolarsLazyWindowAggregation(WindowAggregationFeatureGroup):
    """Polars-based implementation of window aggregation (group-by with broadcast).

    Uses Polars' native .over() window expressions for efficient computation.
    Accepts and returns PyArrow tables (workaround until a dedicated Polars Lazy
    compute framework exists).
    """

    @classmethod
    def compute_framework_rule(cls) -> Union[bool, Set[Type[ComputeFramework]]]:
        return {PyArrowTable}

    @classmethod
    def _compute_window(
        cls,
        table: pa.Table,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
    ) -> pa.Table:
        """Compute a window aggregation using Polars .over() and convert back to PyArrow."""
        # Add a row-order column to preserve original ordering after conversion
        row_order_col = "__polars_row_order__"
        table_with_order = table.append_column(row_order_col, pa.array(range(table.num_rows), type=pa.int64()))

        df = pl.from_arrow(table_with_order)

        if agg_type in _POLARS_AGG_EXPRS:
            expr = _POLARS_AGG_EXPRS[agg_type](source_col).over(partition_by).alias(feature_name)
            df = df.with_columns(expr)
        else:
            df = cls._compute_fallback(df, feature_name, source_col, partition_by, agg_type)

        # Sort by original row order and drop the helper column
        df = df.sort(row_order_col)
        df = df.drop(row_order_col)

        result_arrow = df.to_arrow()

        # Cast the new column to match expected Python types
        result_arrow = cls._cast_result_column(result_arrow, feature_name, agg_type)

        return result_arrow

    @classmethod
    def _cast_result_column(
        cls,
        table: pa.Table,
        feature_name: str,
        agg_type: str,
    ) -> pa.Table:
        """Cast result column types to match expected output conventions."""
        col = table.column(feature_name)
        col_type = col.type

        if agg_type == "avg":
            # Avg should always return float64
            if col_type != pa.float64():
                col = col.cast(pa.float64())
                col_idx = table.column_names.index(feature_name)
                table = table.set_column(col_idx, feature_name, col)
        elif agg_type == "count":
            # Count should return int values; Polars count returns UInt32
            if col_type != pa.int64():
                col = col.cast(pa.int64())
                col_idx = table.column_names.index(feature_name)
                table = table.set_column(col_idx, feature_name, col)

        return table

    @classmethod
    def _compute_fallback(
        cls,
        df: pl.DataFrame,
        feature_name: str,
        source_col: str,
        partition_by: list[str],
        agg_type: str,
    ) -> pl.DataFrame:
        """Fallback for aggregation types not directly supported by Polars .over()."""
        if agg_type == "std":
            expr = pl.col(source_col).std().over(partition_by).alias(feature_name)
        elif agg_type == "var":
            expr = pl.col(source_col).var().over(partition_by).alias(feature_name)
        elif agg_type == "median":
            expr = pl.col(source_col).median().over(partition_by).alias(feature_name)
        elif agg_type == "nunique":
            expr = pl.col(source_col).n_unique().over(partition_by).alias(feature_name)
        elif agg_type == "first":
            expr = pl.col(source_col).first().over(partition_by).alias(feature_name)
        elif agg_type == "last":
            expr = pl.col(source_col).last().over(partition_by).alias(feature_name)
        elif agg_type == "mode":
            expr = pl.col(source_col).mode().first().over(partition_by).alias(feature_name)
        else:
            raise ValueError(f"Unsupported aggregation type: {agg_type}")

        return df.with_columns(expr)
