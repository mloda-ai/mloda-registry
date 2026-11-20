"""Tests for PyArrowBinning compute implementation."""

from __future__ import annotations

import warnings
from typing import Any

import pyarrow as pa

from mloda.community.feature_groups.data_operations.row_preserving.binning.pyarrow_binning import (
    PyArrowBinning,
)
from mloda.testing.feature_groups.data_operations.helpers import calculate_with, make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.frameworks import PyArrowTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.binning.binning import (
    EXPECTED_QBIN_3,
    BinningTestBase,
)


class TestPyArrowBinning(PyArrowTestMixin, BinningTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowBinning

    def test_qbin_no_future_warning_with_nulls(self) -> None:
        """qbin with a null value present must not raise pyarrow's null_placement FutureWarning."""
        fs = make_feature_set("value_int__qbin_3")
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            result = calculate_with(self.implementation_class(), self.test_data, fs)

        result_col = self.extract_column(result, "value_int__qbin_3")
        assert result_col == EXPECTED_QBIN_3

    @staticmethod
    def _bin_column(values: list[Any], arrow_type: pa.DataType, feature: str) -> pa.ChunkedArray:
        table = pa.table({"x": pa.array(values, type=arrow_type)})
        result = calculate_with(PyArrowBinning, table, make_feature_set(feature))
        return result.column(feature)

    def test_qbin_empty_column_gives_empty_int64(self) -> None:
        column = self._bin_column([], pa.int64(), "x__qbin_3")
        assert column.to_pylist() == []
        assert column.type == pa.int64()

    def test_bin_int8_wide_range(self) -> None:
        """int8 -100..100 must not overflow: width 200 / 3 is computed in float64."""
        assert self._bin_column([-100, 0, 100], pa.int8(), "x__bin_3").to_pylist() == [0, 1, 2]

    def test_bin_float32_matches_float64_formula(self) -> None:
        """The float32 values widened to float64: (100.0953369 + 0.0279102) / 25.030814 is 3.99999965, so bin 3."""
        column = self._bin_column([-0.027910225, 125.12616, 100.09534], pa.float32(), "x__bin_5")
        assert column.to_pylist() == [0, 4, 3]
