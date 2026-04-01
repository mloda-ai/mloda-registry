"""Tests for PolarsLazyColumnAggregation compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("polars")

from mloda.community.feature_groups.data_operations.aggregation.polars_lazy_aggregation import (
    PolarsLazyColumnAggregation,
)
from mloda.testing.feature_groups.data_operations.aggregation import (
    AggregationTestBase,
)
from mloda.testing.feature_groups.data_operations.helpers import PolarsLazyTestMixin


class TestPolarsLazyColumnAggregation(PolarsLazyTestMixin, AggregationTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyColumnAggregation
