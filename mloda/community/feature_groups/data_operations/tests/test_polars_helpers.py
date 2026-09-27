"""Tests for the Polars float-only NaN-to-null wrap helper."""

from __future__ import annotations

import pytest

pl = pytest.importorskip("polars")

from mloda.community.feature_groups.data_operations.polars_helpers import nan_to_null


class TestNanToNull:
    def test_float_dtype_fills_nan_with_null(self) -> None:
        """A float column gets .fill_nan(None), so NaN counts as null in aggregations."""
        df = pl.DataFrame({"val": [1.0, float("nan"), 2.0]})
        result = df.select(nan_to_null(pl.col("val"), pl.Float64()).alias("val")).to_series().to_list()
        assert result[0] == 1.0
        assert result[1] is None
        assert result[2] == 2.0

    def test_non_float_dtype_passes_through_unchanged(self) -> None:
        """A non-float column (e.g. Int64) is returned unchanged."""
        df = pl.DataFrame({"n": [1, 2, 3]})
        result = df.select(nan_to_null(pl.col("n"), pl.Int64()).alias("n")).to_series().to_list()
        assert result == [1, 2, 3]
