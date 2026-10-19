"""Tests for DuckdbLookup compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("duckdb")

from mloda.community.feature_groups.data_operations.row_preserving.lookup.duckdb_lookup import (
    DuckdbLookup,
)
from mloda.testing.feature_groups.data_operations.mixins.duckdb import DuckdbTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.lookup.lookup import (
    EmptyMappingHook,
    LookupTestBase,
    NutsMappingHook,
)


class DuckdbNutsLookup(NutsMappingHook, DuckdbLookup):
    """Hook subclass: matches ``*__nuts2024`` by name and resolves its own mapping."""


class _EmptyMappingLookup(EmptyMappingHook, DuckdbLookup):
    """Hook subclass whose resolved mapping is empty (invalid)."""


class TestDuckdbLookup(DuckdbTestMixin, LookupTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbLookup

    @classmethod
    def untyped_null_source_becomes_int(cls) -> bool:
        return True

    @classmethod
    def hook_implementation_class(cls) -> Any:
        return DuckdbNutsLookup

    @classmethod
    def invalid_hook_implementation_class(cls) -> Any:
        return _EmptyMappingLookup
