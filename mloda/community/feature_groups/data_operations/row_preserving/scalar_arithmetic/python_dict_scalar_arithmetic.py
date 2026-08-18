"""PythonDict implementation for single-column element-wise scalar arithmetic."""

from __future__ import annotations

from typing import Any

from mloda.provider import ComputeFramework
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from mloda.community.feature_groups.data_operations.errors import unsupported_op_error
from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
    ARITHMETIC_OPERATIONS,
    ScalarArithmeticFeatureGroup,
)


class PythonDictScalarArithmetic(ScalarArithmeticFeatureGroup):
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PythonDictFramework}

    @classmethod
    def _input_columns_and_framework(cls, data: dict[str, list[Any]]) -> tuple[list[str], str]:
        return list(data.keys()), "PythonDict"

    @classmethod
    def _non_numeric_descriptor(cls, data: dict[str, list[Any]], source_col: str) -> object | None:
        """Return a type-name descriptor when ``source_col`` holds non-numeric values.

        Booleans count as NON-numeric even though ``bool`` is a subclass of
        ``int`` in Python. Nulls are skipped since they carry no type
        information; an all-null column is treated as numeric (nothing to
        reject).
        """
        for value in data[source_col]:
            if value is None:
                continue
            if isinstance(value, bool):
                return "bool"
            if isinstance(value, (int, float)):
                continue
            return type(value).__name__
        return None

    @classmethod
    def _compute_arithmetic(
        cls,
        data: dict[str, list[Any]],
        feature_name: str,
        source_col: str,
        op: str,
        constant: int | float,
    ) -> dict[str, list[Any]]:
        data = dict(data)
        col = data[source_col]

        if op == "add":
            result = [None if v is None else v + constant for v in col]
        elif op == "subtract":
            result = [None if v is None else v - constant for v in col]
        elif op == "multiply":
            result = [None if v is None else v * constant for v in col]
        elif op == "divide":
            result = [None if v is None else v / constant for v in col]
        else:
            raise unsupported_op_error(op, ARITHMETIC_OPERATIONS, framework="PythonDict")

        data[feature_name] = result
        return data
