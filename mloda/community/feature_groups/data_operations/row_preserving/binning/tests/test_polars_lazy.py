"""Tests for PolarsLazyBinning compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("polars")

from mloda.user import DataType

from mloda.community.feature_groups.data_operations.row_preserving.binning.polars_lazy_binning import (
    PolarsLazyBinning,
)
from mloda.testing.feature_groups.data_operations.helpers import calculate_with, make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.frameworks import PolarsLazyTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.binning.binning import (
    BinningTestBase,
)
from mloda.testing.typed_data import TypedTestData


class TestPolarsLazyBinning(PolarsLazyTestMixin, BinningTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyBinning

    @pytest.mark.parametrize(
        "data_type, values",
        [(DataType.BOOLEAN, [True, False, True]), (DataType.STRING, ["1", "2", "10"])],
    )
    def test_non_numeric_source_rejected(self, data_type: DataType, values: list[Any]) -> None:
        data = self.create_test_data(TypedTestData({"x": (data_type, values)}))
        with pytest.raises(ValueError, match="integer or floating"):
            calculate_with(self.implementation_class(), data, make_feature_set("x__bin_3"))
