"""Tests for PyArrowPointArithmetic compute implementation."""

from __future__ import annotations

import math
from typing import Any

import pytest

from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.pyarrow_point_arithmetic import (
    PyArrowPointArithmetic,
)
from mloda.testing.feature_groups.data_operations.helpers import calculate_with, extract_column, make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.frameworks import PyArrowTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.point_arithmetic.point_arithmetic import (
    PointArithmeticTestBase,
)


class TestPyArrowPointArithmetic(PyArrowTestMixin, PointArithmeticTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowPointArithmetic


class TestPyArrowDivideSignedZeroAndNan:
    """Divide follows IEEE-754 for signed-zero denominators and NaN numerators."""

    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [(1.0, -0.0, -math.inf), (-1.0, -0.0, math.inf), (math.nan, 0.0, math.nan)],
        ids=["pos_over_neg_zero", "neg_over_neg_zero", "nan_over_zero"],
    )
    def test_divide_matches_ieee(self, a: float, b: float, expected: float) -> None:
        import pyarrow as pa

        table = pa.table({"a": pa.array([a], type=pa.float64()), "b": pa.array([b], type=pa.float64())})
        result = calculate_with(PyArrowPointArithmetic, table, make_feature_set("a&b__divide_point"))
        value = extract_column(result, "a&b__divide_point")[0]

        if math.isnan(expected):
            assert math.isnan(value)
        else:
            assert value == expected
