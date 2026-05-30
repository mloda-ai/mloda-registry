"""Tests for PandasTimeBucketization compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pandas")

import pandas as pd

from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.pandas_time_bucketization import (
    PandasTimeBucketization,
)
from mloda.testing.feature_groups.data_operations.mixins.pandas import PandasTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.time_bucketization.time_bucketization import (
    TimeBucketizationTestBase,
)


class TestPandasTimeBucketization(PandasTestMixin, TimeBucketizationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PandasTimeBucketization


class TestPandasCalendarRoundDtype:
    """Pin output column dtype for calendar-unit round (month/year).

    Background: pd.DateOffset(months=1) arithmetic on a datetime64[ns, tz]
    Series can fall back to object dtype; the cross-framework value tests
    don't catch this because they compare element-wise as Python objects.
    """

    def _build_pandas_data(self) -> pd.DataFrame:
        from mloda.testing.feature_groups.data_operations.row_preserving.time_bucketization.time_bucketization import (
            _BUCKET_TIMESTAMPS,
        )

        return pd.DataFrame({"timestamp": pd.to_datetime(_BUCKET_TIMESTAMPS, utc=True)})

    def test_round_1_month_dtype_is_datetime64_utc(self) -> None:
        from mloda.testing.feature_groups.data_operations.helpers import make_feature_set

        data = self._build_pandas_data()
        fs = make_feature_set("timestamp__round_1_month")
        result = PandasTimeBucketization.calculate_feature(data, fs)
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp__round_1_month"]), (
            f"Expected datetime64 dtype for round_1_month, got {result['timestamp__round_1_month'].dtype!r}"
        )
        # The input is UTC; the output must remain UTC-tz-aware.
        assert str(result["timestamp__round_1_month"].dt.tz) == "UTC"

    def test_round_1_year_dtype_is_datetime64_utc(self) -> None:
        from mloda.testing.feature_groups.data_operations.helpers import make_feature_set

        data = self._build_pandas_data()
        fs = make_feature_set("timestamp__round_1_year")
        result = PandasTimeBucketization.calculate_feature(data, fs)
        assert pd.api.types.is_datetime64_any_dtype(result["timestamp__round_1_year"]), (
            f"Expected datetime64 dtype for round_1_year, got {result['timestamp__round_1_year'].dtype!r}"
        )
        assert str(result["timestamp__round_1_year"].dt.tz) == "UTC"
