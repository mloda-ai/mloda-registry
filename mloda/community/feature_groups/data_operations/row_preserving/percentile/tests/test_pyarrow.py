"""Tests for PyArrowPercentile compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.percentile.pyarrow_percentile import (
    PyArrowPercentile,
)
from mloda.testing.feature_groups.data_operations.row_preserving.percentile.percentile import (
    PercentileTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.pyarrow import PyArrowTestMixin


class TestPyArrowPercentile(PyArrowTestMixin, PercentileTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowPercentile
