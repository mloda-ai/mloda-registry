"""PyArrow implementation for single-column element-wise scalar arithmetic."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.pyarrow.table import PyArrowTable

from mloda.community.feature_groups.data_operations.errors import unsupported_op_error
from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
    ARITHMETIC_OPERATIONS,
    ScalarArithmeticFeatureGroup,
)


class PyArrowScalarArithmetic(ScalarArithmeticFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PyArrowTable}

    @classmethod
    def _compute_arithmetic(
        cls,
        data: pa.Table,
        feature_name: str,
        source_col: str,
        op: str,
        constant: int | float,
    ) -> pa.Table:
        column = data.column(source_col)
        scalar = pa.scalar(constant)

        if op == "add":
            result = pc.add(column, scalar)
        elif op == "subtract":
            result = pc.subtract(column, scalar)
        elif op == "multiply":
            result = pc.multiply(column, scalar)
        elif op == "divide":
            result = pc.divide(column, scalar)
        else:
            raise unsupported_op_error(op, ARITHMETIC_OPERATIONS, framework="PyArrow")

        return data.append_column(feature_name, result)
