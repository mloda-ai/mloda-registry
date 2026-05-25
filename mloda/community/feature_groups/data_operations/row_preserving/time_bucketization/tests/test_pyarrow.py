"""Tests for PyArrowTimeBucketization compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.pyarrow_time_bucketization import (
    PyArrowTimeBucketization,
)
from mloda.testing.feature_groups.data_operations.mixins.pyarrow import PyArrowTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.time_bucketization.time_bucketization import (
    TimeBucketizationTestBase,
)


class TestPyArrowTimeBucketization(PyArrowTestMixin, TimeBucketizationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowTimeBucketization
