"""Tests for PyArrowLookup compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.lookup.pyarrow_lookup import PyArrowLookup
from mloda.testing.feature_groups.data_operations.mixins.pyarrow import PyArrowTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.lookup.lookup import (
    EmptyMappingHook,
    LookupTestBase,
    NutsMappingHook,
)


class PyArrowNutsLookup(NutsMappingHook, PyArrowLookup):
    """Hook subclass: matches ``*__nuts2024`` by name and resolves its own mapping."""


class _EmptyMappingLookup(EmptyMappingHook, PyArrowLookup):
    """Hook subclass whose resolved mapping is empty (invalid)."""


class TestPyArrowLookup(PyArrowTestMixin, LookupTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowLookup

    @classmethod
    def hook_implementation_class(cls) -> Any:
        return PyArrowNutsLookup

    @classmethod
    def invalid_hook_implementation_class(cls) -> Any:
        return _EmptyMappingLookup
