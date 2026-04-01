"""Tests for PandasGroupAggregation compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pandas")

from mloda.community.feature_groups.data_operations.group_aggregation.pandas_group_aggregation import (
    PandasGroupAggregation,
)
from mloda.testing.feature_groups.data_operations.group_aggregation.group_aggregation import (
    GroupAggregationTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.pandas import PandasTestMixin


class TestPandasGroupAggregation(PandasTestMixin, GroupAggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def supported_agg_types(cls) -> set[str]:
        return {"sum", "avg", "count", "min", "max"}

    @classmethod
    def implementation_class(cls) -> Any:
        return PandasGroupAggregation
