"""Tests for the Polars float-only NaN-to-null wrap helper."""

from __future__ import annotations

import math
from typing import Any

import pytest

pl = pytest.importorskip("polars")

from mloda.community.feature_groups.data_operations.polars_helpers import nan_skipping_extreme, nan_to_null


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


# ---------------------------------------------------------------------------
# nan_skipping_extreme: rolling_min/rolling_max should skip NaN like pc.min/pc.max,
# an all-NaN window stays NaN. See
# docs/guides/data-operation-patterns/03-reference-implementation.md.
# ---------------------------------------------------------------------------


def _rolling(expr_fn: Any, values: list[Any], dtype: Any = pl.Float64()) -> list[Any]:
    df = pl.DataFrame({"val": pl.Series(values, dtype=dtype)})
    expr = nan_skipping_extreme(expr_fn, pl.col("val"), dtype)
    result: list[Any] = df.select(expr.alias("result")).to_series().to_list()
    return result


class TestNanSkippingExtreme:
    def test_rolling_max_skips_nan(self) -> None:
        """rolling_max(window=2) over [1.0, nan, 2.0]: the nan-window skips the NaN peer."""
        fn = lambda col: col.rolling_max(window_size=2, min_samples=1)  # noqa: E731
        result = _rolling(fn, [1.0, float("nan"), 2.0])
        assert result[0] == 1.0
        assert result[1] == 1.0
        assert result[2] == 2.0

    def test_rolling_min_skips_nan(self) -> None:
        """rolling_min(window=2) over [2.0, nan, 1.0]: the nan-window skips the NaN peer."""
        fn = lambda col: col.rolling_min(window_size=2, min_samples=1)  # noqa: E731
        result = _rolling(fn, [2.0, float("nan"), 1.0])
        assert result[0] == 2.0
        assert result[1] == 2.0
        assert result[2] == 1.0

    def test_rolling_max_all_nan_window_stays_nan(self) -> None:
        fn = lambda col: col.rolling_max(window_size=2, min_samples=1)  # noqa: E731
        result = _rolling(fn, [float("nan"), float("nan")])
        assert all(v is not None and math.isnan(v) for v in result)

    def test_rolling_max_null_and_nan_window_stays_nan(self) -> None:
        fn = lambda col: col.rolling_max(window_size=2, min_samples=1)  # noqa: E731
        result = _rolling(fn, [None, float("nan")])
        assert result[1] is not None and math.isnan(result[1])

    def test_rolling_max_all_null_window_stays_null(self) -> None:
        fn = lambda col: col.rolling_max(window_size=2, min_samples=1)  # noqa: E731
        result = _rolling(fn, [None, None])
        assert all(v is None for v in result)

    def test_int64_passthrough(self) -> None:
        """A non-float dtype is passed straight to expr_fn, unwrapped."""
        fn = lambda col: col.rolling_max(window_size=2, min_samples=1)  # noqa: E731
        result = _rolling(fn, [1, 2, 3], dtype=pl.Int64())
        assert result == [1, 2, 3]
