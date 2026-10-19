"""Tests for SqliteLookup compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.lookup.sqlite_lookup import (
    SqliteLookup,
)
from mloda.testing.feature_groups.data_operations.mixins.sqlite import SqliteTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.lookup.lookup import (
    EmptyMappingHook,
    LookupTestBase,
    NutsMappingHook,
)


class SqliteNutsLookup(NutsMappingHook, SqliteLookup):
    """Hook subclass: matches ``*__nuts2024`` by name and resolves its own mapping."""


class _EmptyMappingLookup(EmptyMappingHook, SqliteLookup):
    """Hook subclass whose resolved mapping is empty (invalid)."""


class TestSqliteLookup(SqliteTestMixin, LookupTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteLookup

    @classmethod
    def hook_implementation_class(cls) -> Any:
        return SqliteNutsLookup

    @classmethod
    def invalid_hook_implementation_class(cls) -> Any:
        return _EmptyMappingLookup
