"""Tests for SqliteDateTimeExtraction compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.datetime.sqlite_datetime import (
    SqliteDateTimeExtraction,
)
from mloda.testing.feature_groups.data_operations.mixins.reserved_columns import ReservedColumnsTestMixin
from mloda.testing.feature_groups.data_operations.mixins.sqlite import SqliteTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.datetime.datetime import (
    DateTimeTestBase,
)


class TestSqliteDateTimeExtraction(ReservedColumnsTestMixin, SqliteTestMixin, DateTimeTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteDateTimeExtraction

    @classmethod
    def reserved_columns_feature_name(cls) -> str:
        return "timestamp__year"

    @classmethod
    def reserved_columns_partition_by(cls) -> list[str] | None:
        return None

    @classmethod
    def reserved_columns_order_by(cls) -> str | None:
        return None
