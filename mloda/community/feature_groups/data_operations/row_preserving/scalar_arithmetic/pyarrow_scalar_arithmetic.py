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
    def _input_columns_and_framework(cls, data: pa.Table) -> tuple[list[str], str]:
        return list(data.column_names), "PyArrow"

    @classmethod
    def _assert_source_column_is_numeric(cls, data: pa.Table, source_col: str) -> None:
        arrow_type = data.column(source_col).type
        if pa.types.is_boolean(arrow_type) or not (pa.types.is_integer(arrow_type) or pa.types.is_floating(arrow_type)):
            cls._raise_non_numeric_source(source_col, arrow_type)

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
            result = pc.divide(pc.cast(column, pa.float64()), scalar)
        else:
            raise unsupported_op_error(op, ARITHMETIC_OPERATIONS, framework="PyArrow")

        return data.append_column(feature_name, result)
