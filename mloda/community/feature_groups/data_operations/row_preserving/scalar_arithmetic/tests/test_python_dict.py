"""Tests for PythonDictScalarArithmetic compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.python_dict_scalar_arithmetic import (
    PythonDictScalarArithmetic,
)
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.scalar_arithmetic.scalar_arithmetic import (
    ScalarArithmeticTestBase,
)


class TestPythonDictScalarArithmetic(PythonDictTestMixin, ScalarArithmeticTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictScalarArithmetic
