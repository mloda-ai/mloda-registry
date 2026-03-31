"""Tests for PyArrowWindowAggregation compute implementation."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.pyarrow_window_aggregation import (
    PyArrowWindowAggregation,
)
from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation import (
    WindowAggregationTestBase,
)


class TestPyArrowWindowAggregation(WindowAggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowWindowAggregation

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        return arrow_table

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        return list(result.column(column_name).to_pylist())

    def get_row_count(self, result: Any) -> int:
        return int(result.num_rows)

    def get_expected_type(self) -> Any:
        return pa.Table


class TestPyArrowWindowAggregationNullSorting:
    """Regression tests for null order_by sorting."""

    def test_multiple_null_order_by_first(self) -> None:
        """Two or more null order_by values must not crash when using first aggregation."""
        table = pa.table(
            {
                "region": ["A", "A", "A", "A"],
                "ts": [None, 1, None, 2],
                "value": [100, 10, 200, 20],
            }
        )
        feature = Feature(
            "value__first_groupby",
            options=Options(context={"partition_by": ["region"], "order_by": "ts"}),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = PyArrowWindowAggregation.calculate_feature(table, fs)
        col = result.column("value__first_groupby").to_pylist()

        # first with order_by sorts by ts: [1->10, 2->20, None->100, None->200]
        # first non-null = 10, broadcast to all rows
        assert col == [10, 10, 10, 10]

    def test_multiple_null_order_by_last(self) -> None:
        """Two or more null order_by values must not crash when using last aggregation."""
        table = pa.table(
            {
                "region": ["A", "A", "A", "A"],
                "ts": [None, 1, None, 2],
                "value": [100, 10, 200, 20],
            }
        )
        feature = Feature(
            "value__last_groupby",
            options=Options(context={"partition_by": ["region"], "order_by": "ts"}),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = PyArrowWindowAggregation.calculate_feature(table, fs)
        col = result.column("value__last_groupby").to_pylist()

        # last with order_by sorts by ts: [1->10, 2->20, None->100, None->200]
        # last non-null = 200, broadcast to all rows
        assert col == [200, 200, 200, 200]
