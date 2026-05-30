"""Tests for PolarsLazyTimeBucketization compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("polars")

from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.polars_lazy_time_bucketization import (
    PolarsLazyTimeBucketization,
)
from mloda.testing.feature_groups.data_operations.mixins.polars_lazy import PolarsLazyTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.time_bucketization.time_bucketization import (
    TimeBucketizationTestBase,
)


class TestPolarsLazyTimeBucketization(PolarsLazyTestMixin, TimeBucketizationTestBase):
    """All tests inherited from the base class.

    Mirrors datetime polars_lazy: no ``ReservedColumnsTestMixin`` because the
    base ``calculate_feature`` does not call ``assert_no_reserved_columns``;
    only the SQLite backend's compute does (mirroring sqlite_datetime).
    """

    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyTimeBucketization
