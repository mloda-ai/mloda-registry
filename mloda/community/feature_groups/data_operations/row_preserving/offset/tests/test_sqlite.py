"""Tests for SqliteOffset compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.offset.sqlite_offset import SqliteOffset
from mloda.testing.feature_groups.data_operations.mixins.sqlite import SqliteTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.offset.offset import (
    OffsetTestBase,
)


class TestSqliteOffset(SqliteTestMixin, OffsetTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteOffset
