"""Tests for the SqliteLookup Implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.lookup.sqlite_lookup import (
    SqliteLookup,
)
from mloda.community.feature_groups.data_operations.row_preserving.lookup.tests.test_base import (
    _EmptyMappingHookLookup,
    _HookLookup,
)
from mloda.testing.feature_groups.data_operations.mixins.sqlite import SqliteTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.lookup.lookup import (
    LookupTestBase,
)


class TestSqliteLookup(SqliteTestMixin, LookupTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteLookup

    @classmethod
    def hook_group_class(cls) -> Any:
        return _HookLookup

    @classmethod
    def invalid_hook_group_class(cls) -> Any:
        return _EmptyMappingHookLookup
