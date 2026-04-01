"""Tests for PyArrowColumnAggregation compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.aggregation.pyarrow_aggregation import (
    PyArrowColumnAggregation,
)
from mloda.testing.feature_groups.data_operations.aggregation import (
    AggregationTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.pyarrow import PyArrowTestMixin


class TestPyArrowColumnAggregation(PyArrowTestMixin, AggregationTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowColumnAggregation
