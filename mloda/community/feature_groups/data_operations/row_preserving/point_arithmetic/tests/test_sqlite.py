"""Tests for SqlitePointArithmetic compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.sqlite_point_arithmetic import (
    SqlitePointArithmetic,
)
from mloda.testing.feature_groups.data_operations.row_preserving.point_arithmetic.point_arithmetic import (
    PointArithmeticTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.sqlite import SqliteTestMixin


class TestSqlitePointArithmetic(SqliteTestMixin, PointArithmeticTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return SqlitePointArithmetic

    @classmethod
    def detects_non_numeric_source(cls) -> set[str]:
        # SqliteRelation.from_arrow stores boolean columns with SQLite INTEGER
        # affinity (see mloda_plugins ``_arrow_type_to_sqlite``), so a boolean
        # source column is indistinguishable from int64 at the relation level.
        # We can still reject TEXT-affinity (string) columns.
        return {"string"}

    @classmethod
    def divide_zero_propagates_inf(cls) -> bool:
        # SQLite has no IEEE-754 storage; divide-by-zero rows yield NULL.
        return False
