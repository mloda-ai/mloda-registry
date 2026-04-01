"""Tests for PandasScalarAggregate compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pandas")

from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.pandas_scalar_aggregate import (
    PandasScalarAggregate,
)
from mloda.testing.feature_groups.data_operations.row_preserving.scalar_aggregate.scalar_aggregate import (
    ScalarAggregateTestBase,
)
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.pandas import PandasTestMixin


class TestPandasScalarAggregate(PandasTestMixin, ScalarAggregateTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PandasScalarAggregate

    def test_all_null_column_sum(self) -> None:
        """Known divergence: Pandas sum(all-null) returns 0 (identity element convention)."""
        fs = make_feature_set("score__sum_scalar")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_col = self.extract_column(result, "score__sum_scalar")
        assert all(v == 0 for v in result_col)
