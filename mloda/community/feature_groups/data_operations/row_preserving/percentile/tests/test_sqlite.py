"""Tests for SqlitePercentile compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.percentile.sqlite_percentile import (
    SqlitePercentile,
)
from mloda.testing.feature_groups.data_operations.row_preserving.percentile.percentile import (
    PercentileTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.sqlite import SqliteTestMixin


class TestSqlitePercentile(SqliteTestMixin, PercentileTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return SqlitePercentile
