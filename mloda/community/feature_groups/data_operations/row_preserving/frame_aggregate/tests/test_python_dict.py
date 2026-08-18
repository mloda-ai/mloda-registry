"""Tests for PythonDict frame aggregate implementation.

Uses the unified FrameAggregateTestBase.
"""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.frame_aggregate.frame_aggregate import (
    FrameAggregateTestBase,
    time_frame_options,
)

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.python_dict_frame_aggregate import (
    PythonDictFrameAggregate,
)


class TestPythonDictFrameAggregate(CapabilityHookTestMixin, PythonDictTestMixin, FrameAggregateTestBase):
    """Unified tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictFrameAggregate

    @classmethod
    def capability_supported(cls) -> tuple[tuple[str, Options], ...]:
        return (
            ("value_time_frame", time_frame_options("month")),
            ("value__median_rolling_3", Options()),
        )
