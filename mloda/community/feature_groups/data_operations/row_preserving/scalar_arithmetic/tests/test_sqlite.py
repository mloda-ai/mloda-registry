"""Tests for SqliteScalarArithmetic compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.sqlite_scalar_arithmetic import (
    SqliteScalarArithmetic,
)
from mloda.testing.feature_groups.data_operations.row_preserving.scalar_arithmetic.scalar_arithmetic import (
    ScalarArithmeticTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.sqlite import SqliteTestMixin


class TestSqliteScalarArithmetic(SqliteTestMixin, ScalarArithmeticTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteScalarArithmetic
