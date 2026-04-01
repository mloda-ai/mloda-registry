"""Tests for PolarsLazyFilteredAggregation compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("polars")

from mloda.community.feature_groups.data_operations.row_preserving.filtered_aggregation.polars_lazy_filtered_aggregation import (
    PolarsLazyFilteredAggregation,
)
from mloda.testing.feature_groups.data_operations.mixins.polars_lazy import PolarsLazyTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.filtered_aggregation.filtered_aggregation import (
    FilteredAggregationTestBase,
)


class TestPolarsLazyFilteredAggregation(PolarsLazyTestMixin, FilteredAggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyFilteredAggregation
