"""Tests for DuckdbColumnAggregation compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("duckdb")

from mloda.community.feature_groups.data_operations.aggregation.duckdb_aggregation import (
    DuckdbColumnAggregation,
)
from mloda.testing.feature_groups.data_operations.aggregation.aggregation import (
    AggregationTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.duckdb import DuckdbTestMixin


class TestDuckdbColumnAggregation(DuckdbTestMixin, AggregationTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbColumnAggregation
