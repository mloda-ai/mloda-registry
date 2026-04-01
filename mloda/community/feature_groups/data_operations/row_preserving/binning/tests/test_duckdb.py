"""Tests for DuckdbBinning compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

duckdb = pytest.importorskip("duckdb")

from mloda.community.feature_groups.data_operations.row_preserving.binning.duckdb_binning import (
    DuckdbBinning,
)
from mloda.testing.feature_groups.data_operations.helpers import DuckdbTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.binning import (
    BinningTestBase,
)


class TestDuckdbBinning(DuckdbTestMixin, BinningTestBase):
    """Standard tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbBinning
