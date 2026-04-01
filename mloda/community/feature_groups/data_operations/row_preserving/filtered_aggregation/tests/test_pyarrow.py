"""Tests for PyArrowFilteredAggregation compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.filtered_aggregation.pyarrow_filtered_aggregation import (
    PyArrowFilteredAggregation,
)
from mloda.testing.feature_groups.data_operations.mixins.pyarrow import PyArrowTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.filtered_aggregation.filtered_aggregation import (
    FilteredAggregationTestBase,
)


class TestPyArrowFilteredAggregation(PyArrowTestMixin, FilteredAggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowFilteredAggregation
