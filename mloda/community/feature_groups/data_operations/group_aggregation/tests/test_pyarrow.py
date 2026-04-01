"""Tests for PyArrowGroupAggregation compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.group_aggregation.pyarrow_group_aggregation import (
    PyArrowGroupAggregation,
)
from mloda.testing.feature_groups.data_operations.group_aggregation.group_aggregation import (
    GroupAggregationTestBase,
)
from mloda.testing.feature_groups.data_operations.helpers import PyArrowTestMixin


class TestPyArrowGroupAggregation(PyArrowTestMixin, GroupAggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowGroupAggregation
