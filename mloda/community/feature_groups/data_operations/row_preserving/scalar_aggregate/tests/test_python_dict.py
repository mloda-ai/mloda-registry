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
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictScalarAggregate

    @classmethod
    def capability_supported(cls) -> tuple[tuple[str, Options], ...]:
        return (("value__median_scalar", Options()),)
