"""Tests for PyArrowStringOps compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.string.pyarrow_string import (
    PyArrowStringOps,
)
from mloda.testing.feature_groups.data_operations.mixins.pyarrow import PyArrowTestMixin
from mloda.testing.feature_groups.data_operations.string.string import (
    StringTestBase,
)


class TestPyArrowStringOps(PyArrowTestMixin, StringTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowStringOps
