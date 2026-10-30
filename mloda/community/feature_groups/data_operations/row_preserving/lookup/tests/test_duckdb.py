"""Tests for the DuckdbLookup Implementation."""

from __future__ import annotations

from typing import Any

import pytest

duckdb = pytest.importorskip("duckdb")

from mloda.community.feature_groups.data_operations.row_preserving.lookup.duckdb_lookup import (
    DuckdbLookup,
)
from mloda.community.feature_groups.data_operations.row_preserving.lookup.tests.test_base import (
    _EmptyMappingHookLookup,
    _HookLookup,
)
from mloda.testing.feature_groups.data_operations.mixins.duckdb import DuckdbTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.lookup.lookup import (
    LookupTestBase,
)


class TestDuckdbLookup(DuckdbTestMixin, LookupTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbLookup

    @classmethod
    def hook_group_class(cls) -> Any:
        return _HookLookup

    @classmethod
    def invalid_hook_group_class(cls) -> Any:
        return _EmptyMappingHookLookup

    @classmethod
    def untyped_null_source_becomes_int(cls) -> bool:
        # DuckDB 1.x turns an Arrow null column into INTEGER; 2.x keeps the NULL type.
        return int(duckdb.__version__.split(".")[0]) < 2
