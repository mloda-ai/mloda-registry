"""Tests for SqliteSessionization compute implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.sessionization.sqlite_sessionization import (
    SqliteSessionization,
)
from mloda.testing.feature_groups.data_operations.mixins.frameworks import SqliteTestMixin, SqliteViewTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.sessionization.sessionization import (
    SessionizationTestBase,
)


class TestSqliteSessionization(SqliteTestMixin, SessionizationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteSessionization

    @classmethod
    def pinned_divergences(cls) -> Mapping[str, Mapping[str, list[Any]]]:
        # NaN and None partition keys merge, changing session boundaries and IDs.
        return {"8-nan_none_keys-ts__sessionize_30_minute": {"ts__sessionize_30_minute": [1, 1, 0, 1, 1]}}


class TestSqliteSessionizationViewInput(SqliteViewTestMixin, TestSqliteSessionization):
    """Runs the Sessionization suite on a view input."""
