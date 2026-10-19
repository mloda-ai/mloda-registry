"""Tests for PolarsLazyLookup compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("polars")

from mloda.community.feature_groups.data_operations.row_preserving.lookup.polars_lazy_lookup import (
    PolarsLazyLookup,
)
from mloda.testing.feature_groups.data_operations.mixins.polars_lazy import PolarsLazyTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.lookup.lookup import (
    EmptyMappingHook,
    LookupTestBase,
    NutsMappingHook,
)


class PolarsLazyNutsLookup(NutsMappingHook, PolarsLazyLookup):
    """Hook subclass: matches ``*__nuts2024`` by name and resolves its own mapping."""


class _EmptyMappingLookup(EmptyMappingHook, PolarsLazyLookup):
    """Hook subclass whose resolved mapping is empty (invalid)."""


class TestPolarsLazyLookup(PolarsLazyTestMixin, LookupTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyLookup

    @classmethod
    def hook_implementation_class(cls) -> Any:
        return PolarsLazyNutsLookup

    @classmethod
    def invalid_hook_implementation_class(cls) -> Any:
        return _EmptyMappingLookup
