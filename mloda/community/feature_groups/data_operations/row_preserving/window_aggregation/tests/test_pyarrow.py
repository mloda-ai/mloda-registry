"""Tests for PyArrowWindowAggregation compute implementation."""

from __future__ import annotations

import math
import warnings
from typing import Any

import pyarrow as pa
import pytest
from mloda.user import Options

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.pyarrow_window_aggregation import (
    PyArrowWindowAggregation,
)
from mloda.testing.feature_groups.data_operations.helpers import calculate_with, make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.frameworks import PyArrowTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation.window_aggregation import (
    EXPECTED_FIRST_BY_REGION,
    EXPECTED_LAST_BY_REGION,
    WindowAggregationTestBase,
)

_ALL_NAN_EXTREMA_CASES = [
    pytest.param([math.nan] * 3, math.nan, math.nan, id="all-nan"),
    pytest.param([math.nan, None, math.nan], math.nan, math.nan, id="nan-and-null"),
    pytest.param([math.nan, math.inf, None], math.inf, math.inf, id="nan-and-positive-infinity"),
    pytest.param([-math.inf, math.nan, math.inf], -math.inf, math.inf, id="both-infinities-and-nan"),
    pytest.param([math.nan, 3.0, -2.0], -2.0, 3.0, id="nan-and-finite"),
    pytest.param([None] * 3, None, None, id="all-null"),
]


class TestPyArrowWindowAggregation(CapabilityHookTestMixin, PyArrowTestMixin, WindowAggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowWindowAggregation

    @classmethod
    def capability_unsupported(cls) -> tuple[tuple[str, Options], ...]:
        return (("value__mode_window", Options()),)

    @classmethod
    def merges_signed_zero_keys(cls) -> bool:
        """PyArrow's group_by splits 0.0/-0.0 partition keys and nunique values."""
        return False

    def test_first_window_no_future_warning(self) -> None:
        """first with partition_by must not raise pyarrow's null_placement FutureWarning."""
        fs = make_feature_set("value_int__first_window", ["region"], order_by="value_int")
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            result = calculate_with(self.implementation_class(), self.test_data, fs)

        result_col = self.extract_column(result, "value_int__first_window")
        assert result_col == EXPECTED_FIRST_BY_REGION

    def test_last_window_no_future_warning(self) -> None:
        """last with partition_by must not raise pyarrow's null_placement FutureWarning."""
        fs = make_feature_set("value_int__last_window", ["region"], order_by="value_int")
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            result = calculate_with(self.implementation_class(), self.test_data, fs)

        result_col = self.extract_column(result, "value_int__last_window")
        assert result_col == EXPECTED_LAST_BY_REGION

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
        data = pa.table({"region": ["A", "A", "B"], "v": pa.array([None, None, None], pa.int64())})
        fs = make_feature_set(f"v__{agg_type}_window", ["region"])
        result = calculate_with(self.implementation_class(), data, fs)
        column = result.column(f"v__{agg_type}_window")
        assert column.type == expected_type
        assert column.null_count == 3

    @pytest.mark.parametrize("arrow_type", [pa.float32(), pa.float64()], ids=["float32", "float64"])
    @pytest.mark.parametrize(("values", "expected_min", "expected_max"), _ALL_NAN_EXTREMA_CASES)
    def test_min_max_all_nan_partition(
        self,
        arrow_type: pa.DataType,
        values: list[float | None],
        expected_min: float | None,
        expected_max: float | None,
    ) -> None:
        """An all-NaN partition broadcasts NaN for min/max (PyArrow before 26 returns +/-inf)."""
        table = pa.table(
            {
                "grp": ["case"] * 3 + ["control"] * 3,
                "value": pa.array([*values, 4.0, math.nan, 1.0], type=arrow_type),
            }
        )
        for operation, expected in (("min", expected_min), ("max", expected_max)):
            name = f"value__{operation}_window"
            result = calculate_with(PyArrowWindowAggregation, table, make_feature_set(name, ["grp"]))
            control = 1.0 if operation == "min" else 4.0
            assert result.column(name).to_pylist() == pytest.approx([expected] * 3 + [control] * 3, nan_ok=True)
