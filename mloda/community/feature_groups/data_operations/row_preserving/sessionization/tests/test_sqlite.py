"""Tests for SqliteSessionization compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.sessionization.sqlite_sessionization import (
    SqliteSessionization,
)
from mloda.testing.feature_groups.data_operations.mixins.sqlite import SqliteTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.sessionization.sessionization import (
    SessionizationTestBase,
)


class TestSqliteSessionization(SqliteTestMixin, SessionizationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteSessionization
