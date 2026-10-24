"""Polars Lazy implementation for single-column element-wise scalar arithmetic."""

from __future__ import annotations

from typing import Any

import polars as pl
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.polars.lazy_dataframe import PolarsLazyDataFrame

from mloda.community.feature_groups.data_operations.errors import unsupported_op_error
from mloda.community.feature_groups.data_operations.row_preserving.arithmetic.polars_mixin import PolarsArithmeticMixin
from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
    ARITHMETIC_OPERATIONS,
    ScalarArithmeticFeatureGroup,
)


class PolarsLazyScalarArithmetic(Implementation[ScalarArithmeticFeatureGroup]):
    feature_group = ScalarArithmeticFeatureGroup
    compute_framework = PolarsLazyDataFrame

    @classmethod
    def calculate(
        cls,
        group: type[ScalarArithmeticFeatureGroup],
        data: Any,
        features: FeatureSet,
        framework: ComputeFramework | None,
    ) -> Any:
        return group.apply_arithmetic(
            data,
            features,
            non_numeric=PolarsArithmeticMixin._non_numeric_descriptor,
            compute_arithmetic=cls._compute_arithmetic,
        )

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
