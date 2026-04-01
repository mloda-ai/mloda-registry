"""Tests for DuckdbStringOps compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

duckdb = pytest.importorskip("duckdb")

from mloda.community.feature_groups.data_operations.string.duckdb_string import (
    DuckdbStringOps,
)
from mloda.testing.feature_groups.data_operations.mixins.duckdb import DuckdbTestMixin
from mloda.testing.feature_groups.data_operations.string import (
    StringTestBase,
)


class TestDuckdbStringOps(DuckdbTestMixin, StringTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbStringOps
