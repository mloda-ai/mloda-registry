"""Tests for PyArrowTimeBucketization compute implementation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pyarrow as pa

from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.pyarrow_time_bucketization import (
    PyArrowTimeBucketization,
)
from mloda.testing.feature_groups.data_operations.helpers import calculate_with, make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.frameworks import PyArrowTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.time_bucketization.time_bucketization import (
    TimeBucketizationTestBase,
)


class TestPyArrowTimeBucketization(PyArrowTestMixin, TimeBucketizationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def rejects_numeric_offset_source(cls) -> bool:
        """PyArrow before 22 cannot handle numeric fixed-offset zones."""
        return bool(pa.cpp_version_info.major < 22)

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowTimeBucketization

    def test_ceil_1_week_on_aligned_monday_advances(self) -> None:
        """An aligned Monday ceils to the next Monday, not to itself."""
        table = pa.table({"ts": pa.array([datetime(2023, 1, 2), datetime(2023, 1, 2, 12)], type=pa.timestamp("us"))})
        result = calculate_with(PyArrowTimeBucketization, table, make_feature_set("ts__ceil_1_week"))
        assert result.column("ts__ceil_1_week").to_pylist() == [datetime(2023, 1, 9), datetime(2023, 1, 9)]
