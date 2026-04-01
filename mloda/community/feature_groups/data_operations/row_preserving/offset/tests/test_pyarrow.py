"""Tests for PyArrowOffset compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.offset.pyarrow_offset import PyArrowOffset
from mloda.testing.feature_groups.data_operations.mixins.pyarrow import PyArrowTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.offset.offset import OffsetTestBase


class TestPyArrowOffset(PyArrowTestMixin, OffsetTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowOffset
