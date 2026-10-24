"""Pandas implementation for two-column element-wise point arithmetic."""

from __future__ import annotations

from typing import Any

import pandas as pd
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda.community.feature_groups.data_operations.errors import unsupported_op_error
from mloda.community.feature_groups.data_operations.row_preserving.arithmetic.pandas_mixin import PandasArithmeticMixin
from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.base import (
    ARITHMETIC_OPERATIONS,
    PointArithmeticFeatureGroup,
)


class PandasPointArithmetic(Implementation[PointArithmeticFeatureGroup]):
    feature_group = PointArithmeticFeatureGroup
    compute_framework = PandasDataFrame

    @classmethod
    def calculate(
        cls,
        group: type[PointArithmeticFeatureGroup],
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
