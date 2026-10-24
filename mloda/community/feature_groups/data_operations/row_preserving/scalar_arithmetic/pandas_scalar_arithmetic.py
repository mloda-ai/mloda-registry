"""Pandas implementation for single-column element-wise scalar arithmetic."""

from __future__ import annotations

from typing import Any

import pandas as pd
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.errors import unsupported_op_error
from mloda.community.feature_groups.data_operations.row_preserving.arithmetic.pandas_mixin import PandasArithmeticMixin
from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
    ARITHMETIC_OPERATIONS,
    ScalarArithmeticFeatureGroup,
)


class PandasScalarArithmetic(Implementation[ScalarArithmeticFeatureGroup]):
    feature_group = ScalarArithmeticFeatureGroup
    compute_framework = PandasDataFrame

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
            non_numeric=PandasArithmeticMixin._non_numeric_descriptor,
            compute_arithmetic=cls._compute_arithmetic,
        )

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
