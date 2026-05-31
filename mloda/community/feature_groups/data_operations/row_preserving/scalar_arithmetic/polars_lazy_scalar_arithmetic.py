"""Polars Lazy implementation for single-column element-wise scalar arithmetic."""

from __future__ import annotations

import polars as pl

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.errors import unsupported_op_error
from mloda.community.feature_groups.data_operations.polars_numeric_source import polars_non_numeric_descriptor
from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
    ARITHMETIC_OPERATIONS,
    ScalarArithmeticFeatureGroup,
)


class PolarsLazyScalarArithmetic(ScalarArithmeticFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PolarsLazyDataFrame}

    @classmethod
    def _input_columns_and_framework(cls, data: pl.LazyFrame) -> tuple[list[str], str]:
        return list(data.collect_schema().names()), "Polars"

    @classmethod
    def _assert_source_column_is_numeric(cls, data: pl.LazyFrame, source_col: str) -> None:
        descriptor = polars_non_numeric_descriptor(data.collect_schema()[source_col])
        if descriptor is not None:
            cls._raise_non_numeric_source(source_col, descriptor)

    @classmethod
    def _compute_arithmetic(
        cls,
        data: pl.LazyFrame,
        feature_name: str,
        source_col: str,
        op: str,
        constant: int | float,
    ) -> pl.LazyFrame:
        col = pl.col(source_col)

        if op == "add":
            expr = col + constant
        elif op == "subtract":
            expr = col - constant
        elif op == "multiply":
            expr = col * constant
        elif op == "divide":
            expr = col / constant
        else:
            raise unsupported_op_error(op, ARITHMETIC_OPERATIONS, framework="Polars")

        return data.with_columns(expr.alias(feature_name))
