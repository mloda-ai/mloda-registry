"""Pandas implementation for single-column element-wise scalar arithmetic."""

from __future__ import annotations

import pandas as pd

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.errors import unsupported_op_error
from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
    ARITHMETIC_OPERATIONS,
    ScalarArithmeticFeatureGroup,
)


class PandasScalarArithmetic(ScalarArithmeticFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame}

    @classmethod
    def _input_columns_and_framework(cls, data: pd.DataFrame) -> tuple[list[str], str]:
        return list(data.columns), "Pandas"

    @classmethod
    def _assert_source_column_is_numeric(cls, data: pd.DataFrame, source_col: str) -> None:
        series = data[source_col]
        if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
            cls._raise_non_numeric_source(source_col, series.dtype)

    @classmethod
    def _compute_arithmetic(
        cls,
        data: pd.DataFrame,
        feature_name: str,
        source_col: str,
        op: str,
        constant: int | float,
    ) -> pd.DataFrame:
        data = data.copy()
        col = data[source_col]

        if op == "add":
            data[feature_name] = col + constant
        elif op == "subtract":
            data[feature_name] = col - constant
        elif op == "multiply":
            data[feature_name] = col * constant
        elif op == "divide":
            data[feature_name] = col / constant
        else:
            raise unsupported_op_error(op, ARITHMETIC_OPERATIONS, framework="Pandas")

        return data
