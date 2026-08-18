"""Tests for PythonDictFfill compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.ffill.python_dict_ffill import (
    PythonDictFfill,
)
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.ffill.ffill import (
    FfillTestBase,
)


class TestPythonDictFfill(PythonDictTestMixin, FfillTestBase):
    """All tests inherited from the base class.

    No overrides expected: PythonDict aims for the same full support level as
    pandas and PyArrow (all ffill semantics: leading nulls stay null, interior
    and trailing nulls are carried forward per partition, row order restored).
    """

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictFfill
