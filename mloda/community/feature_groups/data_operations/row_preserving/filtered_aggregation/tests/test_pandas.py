"""Tests for PandasFilteredAggregation compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pandas")

from mloda.community.feature_groups.data_operations.row_preserving.filtered_aggregation.pandas_filtered_aggregation import (
    PandasFilteredAggregation,
)
from mloda.testing.feature_groups.data_operations.mixins.pandas import PandasTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.filtered_aggregation.filtered_aggregation import (
    FilteredAggregationTestBase,
)


class TestPandasFilteredAggregation(PandasTestMixin, FilteredAggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def supported_agg_types(cls) -> set[str]:
        return {"sum", "avg", "count", "min", "max"}

    @classmethod
    def implementation_class(cls) -> Any:
        return PandasFilteredAggregation
