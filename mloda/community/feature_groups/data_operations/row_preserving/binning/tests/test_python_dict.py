"""Tests for PythonDictBinning compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.user import DataType

from mloda.community.feature_groups.data_operations.row_preserving.binning.python_dict_binning import (
    PythonDictBinning,
)
from mloda.testing.feature_groups.data_operations.helpers import calculate_with, make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.frameworks import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.binning.binning import (
    BinningTestBase,
)
from mloda.testing.typed_data import TypedTestData


class TestPythonDictBinning(PythonDictTestMixin, BinningTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictBinning

    def test_width_is_computed_in_float64(self) -> None:
        """Width is (float(max) - float(min)) / n like the reference; a range beyond 2**53 differs from the exact int width."""
        values = [-5226411881888040, -2066086006351943, -1276004537467919, 304158400300130, 6624810151372325]
        data = self.create_test_data(TypedTestData({"x": (DataType.INT64, values)}))
        result = calculate_with(self.implementation_class(), data, make_feature_set("x__bin_15"))
        assert self.extract_column(result, "x__bin_15") == [0, 4, 5, 7, 14]
