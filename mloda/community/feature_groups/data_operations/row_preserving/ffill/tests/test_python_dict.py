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
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictFfill
