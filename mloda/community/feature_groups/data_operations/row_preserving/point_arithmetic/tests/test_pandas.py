"""Tests for PandasPointArithmetic compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pandas")

from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.pandas_point_arithmetic import (
    PandasPointArithmetic,
)
from mloda.testing.feature_groups.data_operations.mixins.pandas import (
    PandasNullableIntTestMixin,
    PandasTestMixin,
)
from mloda.testing.feature_groups.data_operations.row_preserving.point_arithmetic.point_arithmetic import (
    PointArithmeticTestBase,
)


class TestPandasPointArithmetic(PandasTestMixin, PointArithmeticTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PandasPointArithmetic


class TestPandasPointArithmeticNullableInt(PandasNullableIntTestMixin, TestPandasPointArithmetic):
    """Runs the point_arithmetic suite on nullable pandas int columns."""
