"""Tests for PythonDict window aggregation compute implementation.

Uses the unified WindowAggregationTestBase.

PythonDict aims for FULL support (17/17 aggregation types), matching
Pandas, Polars-lazy, and DuckDB, since it is a pure-Python implementation
with no engine-level window-function limitations. No
``supported_agg_types()`` override beyond the ``mean`` alias is needed
(mirrors the DuckDB/Pandas twins: ``{*ALL_AGG_TYPES, "mean"}``).
"""

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
