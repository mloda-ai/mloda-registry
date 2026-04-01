"""Tests for PyArrowWindowAggregation compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.pyarrow_window_aggregation import (
    PyArrowWindowAggregation,
)
from mloda.testing.feature_groups.data_operations.helpers import PyArrowTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation import (
    WindowAggregationTestBase,
)


class TestPyArrowWindowAggregation(PyArrowTestMixin, WindowAggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowWindowAggregation
