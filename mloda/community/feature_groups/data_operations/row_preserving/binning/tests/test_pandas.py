"""Tests for PandasBinning compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pandas")

from mloda.user import DataType

from mloda.community.feature_groups.data_operations.row_preserving.binning.pandas_binning import (
    PandasBinning,
)
from mloda.testing.feature_groups.data_operations.helpers import calculate_with, make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.frameworks import (
    PandasNullableIntTestMixin,
    PandasTestMixin,
)
from mloda.testing.feature_groups.data_operations.row_preserving.binning.binning import (
    BinningTestBase,
)
from mloda.testing.typed_data import TypedTestData


class TestPandasBinning(PandasTestMixin, BinningTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PandasBinning

    @pytest.mark.parametrize(
        "data_type, values",
        [(DataType.BOOLEAN, [True, False, True]), (DataType.STRING, ["1", "2", "10"])],
    )
    def test_non_numeric_source_rejected(self, data_type: DataType, values: list[Any]) -> None:
        data = self.create_test_data(TypedTestData({"x": (data_type, values)}))
        with pytest.raises(ValueError, match="integer or floating"):
            calculate_with(self.implementation_class(), data, make_feature_set("x__bin_3"))


class TestPandasBinningNullableInt(PandasNullableIntTestMixin, TestPandasBinning):
    """Runs the binning suite on nullable pandas int columns."""
