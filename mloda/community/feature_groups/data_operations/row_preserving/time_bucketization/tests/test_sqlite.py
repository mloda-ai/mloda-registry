"""Tests for SqliteTimeBucketization compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.sqlite_time_bucketization import (
    SqliteTimeBucketization,
)
from mloda.testing.feature_groups.data_operations.mixins.reserved_columns import ReservedColumnsTestMixin
from mloda.testing.feature_groups.data_operations.mixins.sqlite import SqliteTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.time_bucketization.time_bucketization import (
    TimeBucketizationTestBase,
)


class TestSqliteTimeBucketization(ReservedColumnsTestMixin, SqliteTestMixin, TimeBucketizationTestBase):
    """All tests inherited from the base class.

    Uses ``ReservedColumnsTestMixin`` to assert that a ``__mloda_``-prefixed
    user column is accepted: there is no reserved-column guard any more, and
    SQLite time bucketization orders by ``rowid`` without adding a helper
    column, so such inputs are processed normally.
    """

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteTimeBucketization

    @classmethod
    def reserved_columns_feature_name(cls) -> str:
        return "timestamp__floor_1_day"

    @classmethod
    def reserved_columns_partition_by(cls) -> list[str] | None:
        return None
