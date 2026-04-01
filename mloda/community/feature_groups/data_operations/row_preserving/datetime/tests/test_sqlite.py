"""Tests for SqliteDateTimeExtraction compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.datetime.sqlite_datetime import (
    SqliteDateTimeExtraction,
)
from mloda.testing.feature_groups.data_operations.helpers import SqliteTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.datetime import (
    DateTimeTestBase,
)


class TestSqliteDateTimeExtraction(SqliteTestMixin, DateTimeTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteDateTimeExtraction
