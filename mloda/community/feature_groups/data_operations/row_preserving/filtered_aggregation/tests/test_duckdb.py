"""Tests for DuckdbFilteredAggregation compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

duckdb = pytest.importorskip("duckdb")

from mloda.community.feature_groups.data_operations.row_preserving.filtered_aggregation.duckdb_filtered_aggregation import (
    DuckdbFilteredAggregation,
)
from mloda.testing.feature_groups.data_operations.mixins.duckdb import DuckdbTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.filtered_aggregation.filtered_aggregation import (
    FilteredAggregationTestBase,
)


class TestDuckdbFilteredAggregation(DuckdbTestMixin, FilteredAggregationTestBase):
    """Standard tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbFilteredAggregation
