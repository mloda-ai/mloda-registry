"""Tests for PolarsLazyScalarArithmetic compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("polars")

from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.polars_lazy_scalar_arithmetic import (
    PolarsLazyScalarArithmetic,
)
from mloda.testing.feature_groups.data_operations.row_preserving.scalar_arithmetic.scalar_arithmetic import (
    ScalarArithmeticTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.polars_lazy import PolarsLazyTestMixin
from mloda.testing.feature_groups.data_operations.mixins.reserved_columns import ReservedColumnsTestMixin


class TestPolarsLazyScalarArithmetic(ReservedColumnsTestMixin, PolarsLazyTestMixin, ScalarArithmeticTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyScalarArithmetic

    @classmethod
    def reserved_columns_feature_name(cls) -> str:
        return "value_int__add_constant"

    @classmethod
    def reserved_columns_partition_by(cls) -> list[str] | None:
        return None
