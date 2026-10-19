"""Tests for PyArrowLookup compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

from mloda.community.feature_groups.data_operations.row_preserving.lookup.pyarrow_lookup import PyArrowLookup
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.pyarrow import PyArrowTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.lookup.lookup import (
    LookupTestBase,
    NutsMappingHook,
    create_lookup_arrow_table,
)


class PyArrowNutsLookup(NutsMappingHook, PyArrowLookup):
    """Hook subclass: matches ``*__nuts2024`` by name and resolves its own mapping."""


class _EmptyMappingLookup(NutsMappingHook, PyArrowLookup):
    PREFIX_PATTERN = r".*__emptymap$"

    @classmethod
    def resolve_mapping(cls, feature: Any) -> dict[str, str]:
        return {}


class TestPyArrowLookup(PyArrowTestMixin, LookupTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowLookup

    @classmethod
    def hook_implementation_class(cls) -> Any:
        return PyArrowNutsLookup


class TestPyArrowLookupHookMapping:
    def test_invalid_hook_mapping_raises_value_error(self) -> None:
        fs = make_feature_set("product_code__emptymap", on_unmatched="null")
        with pytest.raises(ValueError):
            _EmptyMappingLookup.calculate_feature(create_lookup_arrow_table(), fs)
