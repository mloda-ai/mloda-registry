"""Tests for PyArrowScalarAggregate compute implementation."""

from __future__ import annotations

import math
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

_ALL_NAN_EXTREMA_CASES = [
    pytest.param([math.nan] * 3, math.nan, math.nan, id="all-nan"),
    pytest.param([math.nan, None, math.nan], math.nan, math.nan, id="nan-and-null"),
    pytest.param([math.nan, math.inf, None], math.inf, math.inf, id="nan-and-positive-infinity"),
    pytest.param([-math.inf, math.nan, math.inf], -math.inf, math.inf, id="both-infinities-and-nan"),
    pytest.param([math.nan, 3.0, -2.0], -2.0, 3.0, id="nan-and-finite"),
    pytest.param([None] * 3, None, None, id="all-null"),
]


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

    @pytest.mark.parametrize("arrow_type", [pa.float32(), pa.float64()], ids=["float32", "float64"])
    @pytest.mark.parametrize(("values", "expected_min", "expected_max"), _ALL_NAN_EXTREMA_CASES)
    def test_min_max_all_nan_column(
        self,
        arrow_type: pa.DataType,
        values: list[float | None],
        expected_min: float | None,
        expected_max: float | None,
    ) -> None:
        """An all-NaN column broadcasts NaN for min/max (PyArrow before 26 returns +/-inf)."""
        table = pa.table({"value": pa.array(values, type=arrow_type)})
        for operation, expected in (("min", expected_min), ("max", expected_max)):
            name = f"value__{operation}_scalar"
            result = calculate_with(PyArrowScalarAggregate, table, make_feature_set(name))
            assert result.column(name).to_pylist() == pytest.approx([expected] * 3, nan_ok=True)
