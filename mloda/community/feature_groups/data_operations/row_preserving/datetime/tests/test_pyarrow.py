"""Tests for PyArrowDateTimeExtraction compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.datetime.pyarrow_datetime import (
    PyArrowDateTimeExtraction,
)
from mloda.testing.feature_groups.data_operations.mixins.pyarrow import PyArrowTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.datetime import (
    DateTimeTestBase,
)


class TestPyArrowDateTimeExtraction(PyArrowTestMixin, DateTimeTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowDateTimeExtraction
