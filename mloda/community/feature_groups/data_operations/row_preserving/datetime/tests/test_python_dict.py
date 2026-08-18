"""Tests for PythonDictDateTimeExtraction compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.datetime.python_dict_datetime import (
    PythonDictDateTimeExtraction,
)
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.datetime.datetime import (
    DateTimeTestBase,
)


class TestPythonDictDateTimeExtraction(PythonDictTestMixin, DateTimeTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictDateTimeExtraction
