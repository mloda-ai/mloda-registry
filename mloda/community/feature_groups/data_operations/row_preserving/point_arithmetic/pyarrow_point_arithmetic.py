"""PyArrow implementation for two-column element-wise point arithmetic."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from mloda.community.feature_groups.data_operations.errors import unsupported_op_error
from mloda.community.feature_groups.data_operations.row_preserving.arithmetic.pyarrow_mixin import (
    PyArrowArithmeticMixin,
)
from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.base import (
    ARITHMETIC_OPERATIONS,
    PointArithmeticFeatureGroup,
)


class PyArrowPointArithmetic(PyArrowArithmeticMixin, PointArithmeticFeatureGroup):
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
