"""Tests for DuckdbGroupAggregation compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

duckdb = pytest.importorskip("duckdb")

from mloda.community.feature_groups.data_operations.group_aggregation.duckdb_group_aggregation import (
    DuckdbGroupAggregation,
)
from mloda.testing.feature_groups.data_operations.group_aggregation.group_aggregation import (
    GroupAggregationTestBase,
)
from mloda.testing.feature_groups.data_operations.helpers import DuckdbTestMixin


class TestDuckdbGroupAggregation(DuckdbTestMixin, GroupAggregationTestBase):
    """Standard tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbGroupAggregation
