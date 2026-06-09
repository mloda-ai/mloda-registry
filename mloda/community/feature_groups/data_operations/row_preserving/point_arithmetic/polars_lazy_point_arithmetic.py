"""Polars Lazy implementation for two-column element-wise point arithmetic."""

from __future__ import annotations

import polars as pl

from mloda.community.feature_groups.data_operations.errors import unsupported_op_error
from mloda.community.feature_groups.data_operations.arithmetic.polars_mixin import PolarsArithmeticMixin
from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.base import (
    ARITHMETIC_OPERATIONS,
    PointArithmeticFeatureGroup,
)


class PolarsLazyPointArithmetic(PolarsArithmeticMixin, PointArithmeticFeatureGroup):
    @classmethod
    def _compute_arithmetic(
        cls,
        data: pl.LazyFrame,
        feature_name: str,
        col_a: str,
        col_b: str,
        op: str,
    ) -> pl.LazyFrame:
        a = pl.col(col_a)
        b = pl.col(col_b)

        if op == "add":
            expr = a + b
        elif op == "subtract":
            expr = a - b
        elif op == "multiply":
            expr = a * b
        elif op == "divide":
            expr = a / b
        else:
            raise unsupported_op_error(op, ARITHMETIC_OPERATIONS, framework="Polars")

        return data.with_columns(expr.alias(feature_name))
