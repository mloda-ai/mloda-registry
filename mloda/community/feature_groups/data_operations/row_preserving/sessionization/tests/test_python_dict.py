"""Tests for PythonDictSessionization compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.sessionization.python_dict_sessionization import (
    PythonDictSessionization,
)
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.sessionization.sessionization import (
    SessionizationTestBase,
)


class TestPythonDictSessionization(PythonDictTestMixin, SessionizationTestBase):
    """All tests inherited from the base class.

    No overrides expected: PythonDict, being pure Python, sessionizes native
    ``datetime`` values directly (gap = timedelta subtraction, compared against
    a ``timedelta(seconds=threshold_seconds)``) with no timestamp-resolution
    casting concerns. PythonDict targets the same full support level as
    pandas and PyArrow (every backend computes sessionization natively; there
    is no rejection of supported inputs, per the base-class docstring).
    """

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictSessionization
