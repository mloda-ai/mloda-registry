"""Tests for DuckdbBinning compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

duckdb = pytest.importorskip("duckdb")

from mloda.community.feature_groups.data_operations.row_preserving.binning.duckdb_binning import (
    DuckdbBinning,
)
from mloda.testing.feature_groups.data_operations.mixins.duckdb import DuckdbTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.binning.binning import (
    BinningTestBase,
)


class TestDuckdbBinning(DuckdbTestMixin, BinningTestBase):
    """Standard tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbBinning

    def test_collision_rn(self) -> None:
        """User column named __mloda_rn__ must survive DuckdbBinning qbin."""
        self._run_collision_case("__mloda_rn__")
