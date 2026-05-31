"""Tests for PyArrowSessionization compute implementation."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from mloda.community.feature_groups.data_operations.row_preserving.sessionization.pyarrow_sessionization import (
    PyArrowSessionization,
)
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.pyarrow import PyArrowTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.sessionization.sessionization import (
    SessionizationTestBase,
)


class TestPyArrowSessionization(PyArrowTestMixin, SessionizationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowSessionization

    def test_non_timestamp_order_column_rejected(self) -> None:
        """A non-timestamp (int64) order column must be rejected with a ValueError.

        The PyArrow backend cannot natively sessionize a non-timestamp column;
        casting int64 values to microseconds silently produces wrong gaps. Per
        the CFW backend-rejection rule it must raise a clear ValueError rather
        than compute a fallback.
        """
        table = pa.table(
            {
                "id": pa.array([1, 2, 3], type=pa.int64()),
                "user": pa.array(["A", "A", "A"], type=pa.string()),
                "ts": pa.array([1, 2, 3], type=pa.int64()),
            }
        )
        fs = make_feature_set("ts__sessionize_30_minute", partition_by=["user"], order_by="ts")
        with pytest.raises(ValueError, match=r"(?i)timestamp"):
            PyArrowSessionization.calculate_feature(table, fs)
