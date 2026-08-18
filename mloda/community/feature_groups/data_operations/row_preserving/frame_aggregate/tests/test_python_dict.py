"""Tests for PythonDict frame aggregate implementation.

Uses the unified FrameAggregateTestBase.

PythonDict aims for full support (all frame types, all time units including
month/year, all aggregation types) matching DuckDB and Polars-lazy, since it
is a pure-Python implementation with no engine-level rolling/window
limitations. Month/year calendar arithmetic is implemented with stdlib
``calendar.monthrange``-based day-clamping (no ``python-dateutil`` dependency),
matching ``dateutil.relativedelta`` semantics used by the reference
implementation. See known-divergences.md for why SQLite/Pandas narrow this
set instead; PythonDict is not subject to those engine constraints.
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
    """Unified tests inherited from the base class.

    No overrides of ``supported_frame_types()``, ``supported_time_units()``, or
    ``supports_null_order_in_time_window()``: PythonDict aims for the same full
    support level as DuckDB and Polars-lazy (all four frame types, all seven
    time units, tolerates null order_by in time windows).
    """

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictFrameAggregate

    @classmethod
    def capability_supported(cls) -> tuple[tuple[str, Options], ...]:
        return (
            ("value_time_frame", time_frame_options("month")),
            ("value__median_rolling_3", Options()),
        )
