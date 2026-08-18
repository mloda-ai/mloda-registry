"""Tests for PythonDictOffset compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.offset.python_dict_offset import (
    PythonDictOffset,
)
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.offset.offset import OffsetTestBase


class TestPythonDictOffset(PythonDictTestMixin, OffsetTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictOffset
