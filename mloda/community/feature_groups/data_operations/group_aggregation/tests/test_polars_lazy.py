"""Tests for PolarsLazyGroupAggregation compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("polars")

from mloda.community.feature_groups.data_operations.group_aggregation.polars_lazy_group_aggregation import (
    PolarsLazyGroupAggregation,
)
from mloda.testing.feature_groups.data_operations.group_aggregation.group_aggregation import (
    GroupAggregationTestBase,
)
from mloda.testing.feature_groups.data_operations.helpers import PolarsLazyTestMixin


class TestPolarsLazyGroupAggregation(PolarsLazyTestMixin, GroupAggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyGroupAggregation
