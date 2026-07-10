"""Tests for PyArrowAggregation compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.options import Options
from mloda.community.feature_groups.data_operations.aggregation.pyarrow_aggregation import (
    PyArrowAggregation,
)
from mloda.testing.feature_groups.data_operations.aggregation.aggregation import (
    AggregationTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.pyarrow import PyArrowTestMixin


class TestPyArrowAggregation(CapabilityHookTestMixin, PyArrowTestMixin, AggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowAggregation

    @classmethod
    def capability_supported(cls) -> tuple[tuple[str, Options], ...]:
        return (("value__nunique_agg", Options()),)

    @classmethod
    def capability_unsupported(cls) -> tuple[tuple[str, Options], ...]:
        return (
            ("value__median_agg", Options()),
            ("value__mode_agg", Options()),
        )

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
