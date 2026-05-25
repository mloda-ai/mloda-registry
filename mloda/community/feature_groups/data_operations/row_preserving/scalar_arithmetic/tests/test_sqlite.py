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

    @classmethod
    def detects_non_numeric_source(cls) -> set[str]:
        # SqliteRelation.from_arrow stores boolean columns with SQLite INTEGER
        # affinity (see mloda_plugins ``_arrow_type_to_sqlite``), so a boolean
        # source column is indistinguishable from int64 at the relation level.
        # We can still reject TEXT-affinity (string) columns.
        return {"string"}
