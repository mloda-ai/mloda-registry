"""Tests for PyArrowScalarArithmetic compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.pyarrow_scalar_arithmetic import (
    PyArrowScalarArithmetic,
)
from mloda.testing.feature_groups.data_operations.row_preserving.scalar_arithmetic.scalar_arithmetic import (
    ScalarArithmeticTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.pyarrow import PyArrowTestMixin


class TestPyArrowScalarArithmetic(PyArrowTestMixin, ScalarArithmeticTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowScalarArithmetic
