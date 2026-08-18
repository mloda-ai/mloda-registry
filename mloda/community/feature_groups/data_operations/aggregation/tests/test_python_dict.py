"""Tests for PythonDictAggregation compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.options import Options
from mloda.community.feature_groups.data_operations.aggregation.python_dict_aggregation import (
    PythonDictAggregation,
)
from mloda.testing.feature_groups.data_operations.aggregation.aggregation import (
    AggregationTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin


class TestPythonDictAggregation(CapabilityHookTestMixin, PythonDictTestMixin, AggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictAggregation

    @classmethod
    def supported_agg_types(cls) -> set[str]:
        return {*cls.ALL_AGG_TYPES, "mean"}

    @classmethod
    def capability_supported(cls) -> tuple[tuple[str, Options], ...]:
        return (
            ("value__median_agg", Options()),
            ("value__mode_agg", Options()),
        )
