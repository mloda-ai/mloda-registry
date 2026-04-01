"""Tests for PyArrowBinning compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.binning.pyarrow_binning import (
    PyArrowBinning,
)
from mloda.testing.feature_groups.data_operations.helpers import PyArrowTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.binning import (
    BinningTestBase,
)


class TestPyArrowBinning(PyArrowTestMixin, BinningTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowBinning
