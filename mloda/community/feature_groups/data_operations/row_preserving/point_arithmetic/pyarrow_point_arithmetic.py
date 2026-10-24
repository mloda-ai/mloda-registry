"""PyArrow implementation for two-column element-wise point arithmetic."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.errors import unsupported_op_error
from mloda.community.feature_groups.data_operations.row_preserving.arithmetic.pyarrow_mixin import (
    PyArrowArithmeticMixin,
)
from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.base import (
    ARITHMETIC_OPERATIONS,
    PointArithmeticFeatureGroup,
)


class PyArrowPointArithmetic(Implementation[PointArithmeticFeatureGroup]):
    feature_group = PointArithmeticFeatureGroup
    compute_framework = PyArrowTable

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
            non_numeric=PyArrowArithmeticMixin._non_numeric_descriptor,
            compute_arithmetic=cls._compute_arithmetic,
        )

    @classmethod
    def _compute_arithmetic(
        cls,
        data: pa.Table,
        feature_name: str,
        col_a: str,
        col_b: str,
        op: str,
    ) -> pa.Table:
        column_a = data.column(col_a)
        column_b = data.column(col_b)

        if op == "add":
            result = pc.add(column_a, column_b)
        elif op == "subtract":
            result = pc.subtract(column_a, column_b)
        elif op == "multiply":
            result = pc.multiply(column_a, column_b)
        elif op == "divide":
            a_f = pc.cast(column_a, pa.float64())
            b_f = pc.cast(column_b, pa.float64())
            result = pc.divide(a_f, b_f)
        else:
            raise unsupported_op_error(op, ARITHMETIC_OPERATIONS, framework="PyArrow")

        return data.append_column(feature_name, result)
