"""Tests for PandasBinning compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pd = pytest.importorskip("pandas")

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


@pytest.mark.parametrize("op", ["bin", "qbin"])
def test_complex_source_is_rejected(op: str) -> None:
    """Complex counts as numeric for arithmetic, but binning needs integer or floating (#1596)."""
    with pytest.raises(ValueError, match="Binning needs an integer or floating source column"):
        calculate_with(PandasBinning, pd.DataFrame({"x": [1 + 1j, 2 + 0j]}), make_feature_set(f"x__{op}_2"))


class TestPandasBinning(PandasTestMixin, BinningTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PandasBinning


class TestPandasBinningNullableInt(PandasNullableIntTestMixin, TestPandasBinning):
    """Runs the binning suite on nullable pandas int columns."""
