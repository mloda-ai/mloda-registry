"""Tests for PolarsLazyTimeBucketization compute implementation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

pytest.importorskip("polars")

from mloda.user import DataType
from polars.exceptions import ComputeError

from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.polars_lazy_time_bucketization import (
    PolarsLazyTimeBucketization,
)
from mloda.testing.feature_groups.data_operations.mixins.frameworks import PolarsLazyTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.time_bucketization.time_bucketization import (
    TimeBucketizationTestBase,
)
from mloda.testing.typed_data import TypedTestData


class TestPolarsLazyTimeBucketization(PolarsLazyTestMixin, TimeBucketizationTestBase):
    """All tests inherited from the base class.

    Mirrors datetime polars_lazy: no ``ReservedColumnsTestMixin`` here. There is
    no reserved-column guard on any backend; the SQLite test class carries the
    acceptance mixin for the cross-backend coverage.
    """

    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyTimeBucketization

    @classmethod
    def holds_sub_hour_offset(cls) -> bool:
        return False

    def test_sub_hour_offset_still_unsupported(self) -> None:
        """Flags when Polars can hold a -03:30 offset, so the skip can go."""
        data = TypedTestData(
            {
                "ts": (
                    DataType.TIMESTAMP_MICROS_TZ,
                    [datetime(2024, 1, 1, 2, tzinfo=timezone(timedelta(hours=-3, minutes=-30)))],
                )
            }
        )
        with pytest.raises(ComputeError, match="unable to parse time zone"):
            built = self.create_test_data(data)
            if hasattr(built, "collect"):
                built.collect()
