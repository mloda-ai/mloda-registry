"""Polars Lazy implementation for single-column element-wise scalar arithmetic."""

from __future__ import annotations

from typing import Any

import polars as pl

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.errors import unsupported_op_error
from mloda.community.feature_groups.data_operations.reserved_columns import assert_no_reserved_columns
from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
    ARITHMETIC_OPERATIONS,
    ScalarArithmeticFeatureGroup,
)


class PolarsLazyScalarArithmetic(ScalarArithmeticFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PolarsLazyDataFrame}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        # Reserved-column guard runs before option validation so that the
        # shared ReservedColumnsTestMixin (which does not set a constant)
        # surfaces the reserved-column error, not the missing-constant one.
        assert_no_reserved_columns(data.collect_schema().names(), framework="Polars", operation="scalar arithmetic")
        return super().calculate_feature(data, features)

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
