"""Tests for PythonDict window aggregation compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.options import Options
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.python_dict_window_aggregation import (
    PythonDictWindowAggregation,
)
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation.window_aggregation import (
    WindowAggregationTestBase,
)


class TestPythonDictWindowAggregation(CapabilityHookTestMixin, PythonDictTestMixin, WindowAggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictWindowAggregation

    @classmethod
    def supported_agg_types(cls) -> set[str]:
        return {*cls.ALL_AGG_TYPES, "mean"}

    @classmethod
    def capability_supported(cls) -> tuple[tuple[str, Options], ...]:
        return (
            ("value__median_window", Options()),
            ("value__mode_window", Options()),
        )
