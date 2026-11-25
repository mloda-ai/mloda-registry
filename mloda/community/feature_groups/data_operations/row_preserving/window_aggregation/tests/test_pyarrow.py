"""Tests for PyArrowWindowAggregation compute implementation."""

from __future__ import annotations

import warnings
from typing import Any

import pyarrow as pa
import pytest
from mloda.user import Options

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.pyarrow_window_aggregation import (
    PyArrowWindowAggregation,
)
from mloda.community.feature_groups.data_operations.tests.pyarrow_all_nan_cases import (
    ALL_NAN_EXTREMA_CASES,
    all_nan_case_control_table,
)
from mloda.testing.feature_groups.data_operations.helpers import (
    calculate_with,
    extract_column,
    make_feature_set,
)
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.frameworks import PyArrowTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation.window_aggregation import (
    EXPECTED_FIRST_BY_REGION,
    EXPECTED_LAST_BY_REGION,
    WindowAggregationTestBase,
)


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
    @pytest.mark.parametrize(("values", "expected_min", "expected_max"), ALL_NAN_EXTREMA_CASES)
    def test_min_max_all_nan_partition(
        self,
        arrow_type: pa.DataType,
        values: list[float | None],
        expected_min: float | None,
        expected_max: float | None,
    ) -> None:
        """An all-NaN partition broadcasts NaN for min/max (PyArrow before 26 returns +/-inf)."""
        table = all_nan_case_control_table(values, arrow_type)
        for operation, expected in (("min", expected_min), ("max", expected_max)):
            name = f"value__{operation}_window"
            fs = make_feature_set(name, ["grp"])
            result = calculate_with(PyArrowWindowAggregation, table, fs)
            control = 1.0 if operation == "min" else 4.0
            expected_column = [expected] * 3 + [control] * 3
            assert result.column(name).type == arrow_type
            assert result.column(name).to_pylist() == pytest.approx(expected_column, nan_ok=True)
            reference = self.reference_calculate(table, fs)
            assert extract_column(reference, name) == pytest.approx(expected_column, nan_ok=True)
