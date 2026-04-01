"""Tests for PandasColumnAggregation compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pandas")

from mloda.community.feature_groups.data_operations.aggregation.pandas_aggregation import (
    PandasColumnAggregation,
)
from mloda.testing.feature_groups.data_operations.aggregation import (
    AggregationTestBase,
)
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.pandas import PandasTestMixin


class TestPandasColumnAggregation(PandasTestMixin, AggregationTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PandasColumnAggregation

    def test_all_null_column_sum(self) -> None:
        """Known divergence: Pandas sum(all-null) returns 0 (identity element convention)."""
        fs = make_feature_set("score__sum_aggr")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_col = self.extract_column(result, "score__sum_aggr")
        assert all(v == 0 for v in result_col)
