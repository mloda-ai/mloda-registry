"""Tests for PythonDictScalarAggregate compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.options import Options
from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.python_dict_scalar_aggregate import (
    PythonDictScalarAggregate,
)
from mloda.testing.feature_groups.data_operations.row_preserving.scalar_aggregate.scalar_aggregate import (
    ScalarAggregateTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin


class TestPythonDictScalarAggregate(CapabilityHookTestMixin, PythonDictTestMixin, ScalarAggregateTestBase):
    """All tests inherited from the base class.

    PythonDict aims for full support (all 13 aggregation types, including
    median), matching PyArrow and Pandas, since it is a pure-Python
    implementation with no engine-level type restrictions. No
    ``supported_agg_types()`` override is needed.
    """

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictScalarAggregate

    @classmethod
    def capability_supported(cls) -> tuple[tuple[str, Options], ...]:
        return (("value__median_scalar", Options()),)
