"""Tests for DuckdbRank compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

duckdb = pytest.importorskip("duckdb")

from mloda.community.feature_groups.data_operations.row_preserving.rank.duckdb_rank import (
    DuckdbRank,
)
from mloda.testing.feature_groups.data_operations.mixins.duckdb import DuckdbTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.rank.rank import (
    RankTestBase,
)


class TestDuckdbRank(DuckdbTestMixin, RankTestBase):
    """Standard tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbRank

    def test_collision_orig_rn(self) -> None:
        """User column named __mloda_orig_rn must survive DuckdbRank."""
        self._run_collision_case("__mloda_orig_rn")
