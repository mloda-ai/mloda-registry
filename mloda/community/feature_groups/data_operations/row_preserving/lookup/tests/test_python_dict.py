"""Tests for PythonDictLookup compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.lookup.python_dict_lookup import (
    PythonDictLookup,
)
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.lookup.lookup import (
    EmptyMappingHook,
    LookupTestBase,
    NutsMappingHook,
)


class PythonDictNutsLookup(NutsMappingHook, PythonDictLookup):
    """Hook subclass: matches ``*__nuts2024`` by name and resolves its own mapping."""


class _EmptyMappingLookup(EmptyMappingHook, PythonDictLookup):
    """Hook subclass whose resolved mapping is empty (invalid)."""


class TestPythonDictLookup(PythonDictTestMixin, LookupTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictLookup

    @classmethod
    def hook_implementation_class(cls) -> Any:
        return PythonDictNutsLookup

    @classmethod
    def invalid_hook_implementation_class(cls) -> Any:
        return _EmptyMappingLookup
