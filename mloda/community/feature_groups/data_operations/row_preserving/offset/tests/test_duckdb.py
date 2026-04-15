"""Tests for DuckdbOffset compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

duckdb = pytest.importorskip("duckdb")

from mloda.community.feature_groups.data_operations.row_preserving.offset.duckdb_offset import DuckdbOffset
from mloda.testing.feature_groups.data_operations.mixins.duckdb import DuckdbTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.offset.offset import OffsetTestBase


class TestDuckdbOffset(DuckdbTestMixin, OffsetTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbOffset

    def test_collision_orig_rn(self) -> None:
        """User column named __mloda_orig_rn__ must survive DuckdbOffset."""
        self._run_collision_case("__mloda_orig_rn__")
