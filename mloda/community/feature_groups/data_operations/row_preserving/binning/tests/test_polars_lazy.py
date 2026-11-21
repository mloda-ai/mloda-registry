"""Tests for PolarsLazyBinning compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pl = pytest.importorskip("polars")

from mloda.community.feature_groups.data_operations.row_preserving.binning.polars_lazy_binning import (
    PolarsLazyBinning,
)
from mloda.testing.feature_groups.data_operations.helpers import calculate_with, make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.frameworks import PolarsLazyTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.binning.binning import (
    BinningTestBase,
)


class TestPolarsLazyBinning(PolarsLazyTestMixin, BinningTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyBinning

    def test_bin_uint128_beyond_2_53_rejected(self) -> None:
        """A UInt128 bin source beyond 2**53 raises ValueError."""
        if not hasattr(pl, "UInt128"):
            pytest.skip("polars has no UInt128")
        data = pl.LazyFrame({"x": pl.Series([2**53 + 1, 0], dtype=pl.UInt128)})
        with pytest.raises(ValueError, match=r"within 2\*\*53"):
            calculate_with(self.implementation_class(), data, make_feature_set("x__bin_2"))
