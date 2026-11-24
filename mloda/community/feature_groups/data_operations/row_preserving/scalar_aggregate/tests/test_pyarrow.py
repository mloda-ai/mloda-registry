"""Tests for PyArrowScalarAggregate compute implementation."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.pyarrow_scalar_aggregate import (
    PyArrowScalarAggregate,
)
from mloda.testing.feature_groups.data_operations.helpers import calculate_with, make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.frameworks import PyArrowTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.scalar_aggregate.scalar_aggregate import (
    ScalarAggregateTestBase,
)


class TestPyArrowScalarAggregate(PyArrowTestMixin, ScalarAggregateTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowScalarAggregate

    @pytest.mark.parametrize(
        ("agg_type", "expected_type"),
        [
            ("sum", pa.int64()),
            ("min", pa.int64()),
            ("max", pa.int64()),
            ("avg", pa.float64()),
            ("std", pa.float64()),
            ("var", pa.float64()),
        ],
    )
    def test_all_null_source_keeps_the_aggregate_type(self, agg_type: str, expected_type: pa.DataType) -> None:
        """An all-null int64 column aggregates to the aggregate's type, not null."""
        data = pa.table({"v": pa.array([None, None, None], pa.int64())})
        fs = make_feature_set(f"v__{agg_type}_scalar")
        result = calculate_with(self.implementation_class(), data, fs)
        column = result.column(f"v__{agg_type}_scalar")
        assert column.type == expected_type
        assert column.null_count == 3
