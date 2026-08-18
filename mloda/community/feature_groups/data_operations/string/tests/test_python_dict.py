"""Tests for PythonDictStringOps compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.string.python_dict_string import (
    PythonDictStringOps,
)
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.string.string import (
    StringTestBase,
)


class TestPythonDictStringOps(PythonDictTestMixin, StringTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictStringOps
