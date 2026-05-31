"""Pandas implementation for two-column element-wise point arithmetic."""

from __future__ import annotations

import pandas as pd

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.errors import unsupported_op_error
from mloda.community.feature_groups.data_operations.numeric_source import pandas_non_numeric_descriptor
from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.base import (
    ARITHMETIC_OPERATIONS,
    PointArithmeticFeatureGroup,
)


class PandasPointArithmetic(PointArithmeticFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame}

    @classmethod
    def _input_columns_and_framework(cls, data: pd.DataFrame) -> tuple[list[str], str]:
        return list(data.columns), "Pandas"

    @classmethod
    def _assert_source_column_is_numeric(cls, data: pd.DataFrame, source_col: str) -> None:
        descriptor = pandas_non_numeric_descriptor(data[source_col])
        if descriptor is not None:
            cls._raise_non_numeric_source(source_col, descriptor)

    @classmethod
    def _compute_arithmetic(
        cls,
        data: pd.DataFrame,
        feature_name: str,
        col_a: str,
        col_b: str,
        op: str,
    ) -> pd.DataFrame:
        data = data.copy()

        if op == "add":
            data[feature_name] = data[col_a] + data[col_b]
        elif op == "subtract":
            data[feature_name] = data[col_a] - data[col_b]
        elif op == "multiply":
            data[feature_name] = data[col_a] * data[col_b]
        elif op == "divide":
            data[feature_name] = data[col_a] / data[col_b]
        else:
            raise unsupported_op_error(op, ARITHMETIC_OPERATIONS, framework="Pandas")

        return data
