"""Tests for PyArrowDateTimeExtraction compute implementation."""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from mloda.community.feature_groups.data_operations.row_preserving.datetime.pyarrow_datetime import (
    PyArrowDateTimeExtraction,
)
from mloda.testing.feature_groups.data_operations.mixins.frameworks import PyArrowTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.datetime.datetime import (
    DateTimeTestBase,
)


class TestPyArrowDateTimeExtraction(PyArrowTestMixin, DateTimeTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def rejects_numeric_offset_source(cls) -> bool:
        """PyArrow before 22 cannot handle numeric fixed-offset zones."""
        return bool(pa.cpp_version_info.major < 22)

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowDateTimeExtraction
