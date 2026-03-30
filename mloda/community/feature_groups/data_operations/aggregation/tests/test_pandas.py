"""Tests for PandasColumnAggregation compute implementation."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

pytest.importorskip("pandas")

import pandas as pd

from mloda.community.feature_groups.data_operations.aggregation.pandas_aggregation import (
    PandasColumnAggregation,
)
from mloda.testing.feature_groups.data_operations.aggregation import (
    AggregationTestBase,
)


class TestPandasColumnAggregation(AggregationTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PandasColumnAggregation

    def create_test_data(self, arrow_table: pa.Table) -> Any:
        return arrow_table.to_pandas()

    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        series = result[column_name]
        return [None if pd.isna(v) else v for v in series.tolist()]

    def get_row_count(self, result: Any) -> int:
        return len(result)

    def get_expected_type(self) -> Any:
        return pd.DataFrame
