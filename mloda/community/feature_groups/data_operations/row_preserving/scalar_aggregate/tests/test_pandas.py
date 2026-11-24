"""Tests for PandasScalarAggregate compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pandas")

import pandas as pd
from mloda.user import Options

from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.pandas_scalar_aggregate import (
    PandasScalarAggregate,
)
from mloda.testing.feature_groups.data_operations.helpers import calculate_with, make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.frameworks import (
    PandasNullableIntTestMixin,
    PandasTestMixin,
)
from mloda.testing.feature_groups.data_operations.row_preserving.scalar_aggregate.scalar_aggregate import (
    ScalarAggregateTestBase,
)


class TestPandasScalarAggregate(CapabilityHookTestMixin, PandasTestMixin, ScalarAggregateTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PandasScalarAggregate

    @classmethod
    def capability_supported(cls) -> tuple[tuple[str, Options], ...]:
        return (("value__median_scalar", Options()),)


class TestPandasScalarAggregateNullableInt(PandasNullableIntTestMixin, TestPandasScalarAggregate):
    """Runs the scalar_aggregate suite on nullable pandas int columns."""

    @pytest.mark.parametrize(
        ("agg_type", "expected_dtype"),
        [
            ("sum", "Int64"),
            ("min", "Int64"),
            ("max", "Int64"),
            ("avg", "float64"),
            ("std", "float64"),
            ("var", "float64"),
            ("median", "float64"),
        ],
    )
    def test_all_null_int64_source_keeps_the_aggregate_dtype(self, agg_type: str, expected_dtype: str) -> None:
        """An all-null Int64 column aggregates to the aggregate's dtype, never object."""
        data = pd.DataFrame({"v": pd.Series([None, None, None], dtype="Int64")})
        fs = make_feature_set(f"v__{agg_type}_scalar")
        result = calculate_with(self.implementation_class(), data, fs)
        column = result[f"v__{agg_type}_scalar"]
        assert str(column.dtype) == expected_dtype
        assert column.isna().all()
