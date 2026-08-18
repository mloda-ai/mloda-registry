"""Tests for PythonDictPointArithmetic compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.python_dict_point_arithmetic import (
    PythonDictPointArithmetic,
)
from mloda.testing.feature_groups.data_operations.row_preserving.point_arithmetic.point_arithmetic import (
    PointArithmeticTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin


class TestPythonDictPointArithmetic(PythonDictTestMixin, PointArithmeticTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictPointArithmetic
