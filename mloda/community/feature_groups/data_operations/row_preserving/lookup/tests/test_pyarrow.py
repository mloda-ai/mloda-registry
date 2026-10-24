"""Tests for the PyArrowLookup Implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.lookup.pyarrow_lookup import PyArrowLookup
from mloda.community.feature_groups.data_operations.row_preserving.lookup.tests.test_base import (
    _EmptyMappingHookLookup,
    _HookLookup,
)
from mloda.testing.feature_groups.data_operations.mixins.pyarrow import PyArrowTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.lookup.lookup import (
    LookupTestBase,
)


class TestPyArrowLookup(PyArrowTestMixin, LookupTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowLookup

    @classmethod
    def hook_group_class(cls) -> Any:
        return _HookLookup

    @classmethod
    def invalid_hook_group_class(cls) -> Any:
        return _EmptyMappingHookLookup
