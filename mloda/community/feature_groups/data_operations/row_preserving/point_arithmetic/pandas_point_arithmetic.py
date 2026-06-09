"""Pandas implementation for two-column element-wise point arithmetic."""

from __future__ import annotations

import pandas as pd

from mloda.community.feature_groups.data_operations.errors import unsupported_op_error
from mloda.community.feature_groups.data_operations.arithmetic.pandas_mixin import PandasArithmeticMixin
from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.base import (
    ARITHMETIC_OPERATIONS,
    PointArithmeticFeatureGroup,
)


class PandasPointArithmetic(PandasArithmeticMixin, PointArithmeticFeatureGroup):
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
