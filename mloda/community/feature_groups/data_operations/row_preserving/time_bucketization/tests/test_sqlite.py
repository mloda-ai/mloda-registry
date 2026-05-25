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

    Mirrors datetime sqlite: uses ``ReservedColumnsTestMixin`` because the
    SQLite compute calls ``assert_no_reserved_columns`` (analogous to
    ``sqlite_datetime``).
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
