"""Tests for PolarsLazyScalarArithmetic compute implementation."""

from __future__ import annotations

from typing import Any

import pytest
from mloda.core.abstract_plugins.components.data_types import DataType

pytest.importorskip("polars")

from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.polars_lazy_scalar_arithmetic import (
    PolarsLazyScalarArithmetic,
)
from mloda.testing.feature_groups.data_operations.helpers import append_typed_columns, calculate_with, make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.frameworks import PolarsLazyTestMixin
from mloda.testing.feature_groups.data_operations.mixins.reserved_columns import ReservedColumnsTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.scalar_arithmetic.scalar_arithmetic import (
    ScalarArithmeticTestBase,
)


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

    def test_mixin_reserved_column_collision_accepted(self) -> None:
        """Scalar arithmetic needs a ``constant`` option, which the shared mixin
        does not supply; override to pass one while keeping the acceptance check.

        A ``__mloda_``-prefixed user column is accepted (no reserved namespace
        exists) and processed normally.
        """
        colliding_name = "__MLODA_USER_COL__"
        colliding_typed = append_typed_columns(
            self._source, {colliding_name: (DataType.INT64, [0] * self._source.row_count)}
        )
        colliding_data = self.create_test_data(colliding_typed)
        fs = make_feature_set(self.reserved_columns_feature_name(), constant=5)
        result = calculate_with(self.implementation_class(), colliding_data, fs)
        assert result is not None
