"""PythonDict implementation for two-column element-wise point arithmetic."""

from __future__ import annotations

import math
from typing import Any

from mloda.provider import ComputeFramework, FeatureSet, Implementation
from mloda_plugins.compute_framework.base_implementations.python_dict.python_dict_framework import (
    PythonDictFramework,
)

from mloda.community.feature_groups.data_operations.errors import unsupported_op_error
from mloda.community.feature_groups.data_operations.python_dict_helpers import non_numeric_descriptor
from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.base import (
    ARITHMETIC_OPERATIONS,
    PointArithmeticFeatureGroup,
)


def _div(a: float, b: float) -> float:
    """IEEE-754 divide semantics: N/0 -> signed inf, 0/0 -> nan, nan/x -> nan.

    Python's bare ``/`` raises ``ZeroDivisionError`` on zero denominators,
    unlike PyArrow/Pandas/Polars/DuckDB, which all return IEEE-754 inf/nan.
    PythonDict matches that four-backend majority behavior explicitly here.
    A NaN numerator or denominator always yields nan, checked before the
    zero-divisor branch. The sign of an inf result is ``sign(a) * sign(b)``
    (via ``math.copysign``), which also gets signed-zero denominators
    (``-0.0``) right, not just ``sign(a)`` alone.
    """
    if math.isnan(a) or math.isnan(b):
        return float("nan")
    if b == 0.0:
        if a == 0.0:
            return float("nan")
        return math.copysign(float("inf"), a) * math.copysign(1.0, b)
    return a / b


class PythonDictPointArithmetic(Implementation[PointArithmeticFeatureGroup]):
    feature_group = PointArithmeticFeatureGroup
    compute_framework = PythonDictFramework

    @classmethod
    def calculate(
        cls,
        group: type[PointArithmeticFeatureGroup],
        data: Any,
        features: FeatureSet,
        framework: ComputeFramework | None,
    ) -> Any:
        return group.apply_arithmetic(
            data, features, non_numeric=non_numeric_descriptor, compute_arithmetic=cls._compute_arithmetic
        )

    @classmethod
    def _compute_arithmetic(
        cls,
        data: dict[str, list[Any]],
        feature_name: str,
        col_a: str,
        col_b: str,
        op: str,
    ) -> dict[str, list[Any]]:
        data = dict(data)
        column_a = data[col_a]
        column_b = data[col_b]

        if op == "add":
            result: list[Any] = [None if a is None or b is None else a + b for a, b in zip(column_a, column_b)]
        elif op == "subtract":
            result = [None if a is None or b is None else a - b for a, b in zip(column_a, column_b)]
        elif op == "multiply":
            result = [None if a is None or b is None else a * b for a, b in zip(column_a, column_b)]
        elif op == "divide":
            result = [None if a is None or b is None else _div(float(a), float(b)) for a, b in zip(column_a, column_b)]
        else:
            raise unsupported_op_error(op, ARITHMETIC_OPERATIONS, framework="PythonDict")

        data[feature_name] = result
        return data
