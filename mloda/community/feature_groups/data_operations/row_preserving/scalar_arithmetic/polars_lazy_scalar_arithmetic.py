"""Polars Lazy implementation for single-column element-wise scalar arithmetic."""

from __future__ import annotations

import polars as pl

from mloda.community.feature_groups.data_operations.errors import unsupported_op_error
from mloda.community.feature_groups.data_operations.polars_arithmetic_mixin import PolarsArithmeticMixin
from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
    ARITHMETIC_OPERATIONS,
    ScalarArithmeticFeatureGroup,
)


class PolarsLazyScalarArithmetic(PolarsArithmeticMixin, ScalarArithmeticFeatureGroup):
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
