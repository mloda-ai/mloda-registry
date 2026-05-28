"""Tests for PolarsLazyPointArithmetic compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("polars")

from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.polars_lazy_point_arithmetic import (
    PolarsLazyPointArithmetic,
)
from mloda.testing.feature_groups.data_operations.row_preserving.point_arithmetic.point_arithmetic import (
    PointArithmeticTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.polars_lazy import PolarsLazyTestMixin
from mloda.testing.feature_groups.data_operations.mixins.reserved_columns import ReservedColumnsTestMixin


class TestPolarsLazyPointArithmetic(ReservedColumnsTestMixin, PolarsLazyTestMixin, PointArithmeticTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyPointArithmetic

    @classmethod
    def reserved_columns_feature_name(cls) -> str:
        return "value_int&amount__add_point"

    @classmethod
    def reserved_columns_partition_by(cls) -> list[str] | None:
        return None
