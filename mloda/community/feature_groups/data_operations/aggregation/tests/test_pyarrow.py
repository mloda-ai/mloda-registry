"""Tests for PyArrowAggregation compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.aggregation.pyarrow_aggregation import (
    PyArrowAggregation,
)
from mloda.testing.feature_groups.data_operations.aggregation.aggregation import (
    AggregationTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.pyarrow import PyArrowTestMixin


class TestPyArrowAggregation(PyArrowTestMixin, AggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowAggregation

    @classmethod
    def supported_agg_types(cls) -> set[str]:
        return {
            "sum",
            "avg",
            "mean",
            "count",
            "min",
            "max",
            "std",
            "var",
            "std_pop",
            "std_samp",
            "var_pop",
            "var_samp",
            "nunique",
            "first",
            "last",
        }
