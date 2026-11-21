"""Tests for PyArrowResample compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_changing.resample.pyarrow_resample import (
    PyArrowResample,
)
from mloda.testing.feature_groups.data_operations.mixins.frameworks import PyArrowTestMixin
from mloda.testing.feature_groups.data_operations.row_changing.resample.resample import (
    ResampleTestBase,
)


class TestPyArrowResample(PyArrowTestMixin, ResampleTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowResample

    @classmethod
    def merges_signed_zero_keys(cls) -> bool:
        """PyArrow's own Table.group_by() splits 0.0/-0.0 partition keys."""
        return False
