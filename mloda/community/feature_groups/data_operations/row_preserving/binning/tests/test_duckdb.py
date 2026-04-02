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
    """Standard tests inherited from the base class.

    qbin is excluded: DuckdbRelation lacks a public order() method, so NTILE
    results cannot be re-sorted to original row order without eager materialization.
    Tracked in https://github.com/mloda-ai/mloda/issues/251.
    """

    @classmethod
    def supported_ops(cls) -> set[str]:
        return {"bin"}

    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbBinning
