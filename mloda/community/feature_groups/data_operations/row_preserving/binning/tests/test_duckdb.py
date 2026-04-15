"""Tests for DuckdbBinning compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

duckdb = pytest.importorskip("duckdb")

from mloda.community.feature_groups.data_operations.row_preserving.binning.duckdb_binning import (
    DuckdbBinning,
)
from mloda.testing.feature_groups.data_operations.mixins.duckdb import DuckdbTestMixin
from mloda.testing.feature_groups.data_operations.mixins.reserved_columns import ReservedColumnsTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.binning.binning import (
    BinningTestBase,
)


class TestDuckdbBinning(ReservedColumnsTestMixin, DuckdbTestMixin, BinningTestBase):
    """Standard tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbBinning

    @classmethod
    def reserved_columns_feature_name(cls) -> str:
        return "value_int__bin_3"

    @classmethod
    def reserved_columns_partition_by(cls) -> list[str] | None:
        return None
