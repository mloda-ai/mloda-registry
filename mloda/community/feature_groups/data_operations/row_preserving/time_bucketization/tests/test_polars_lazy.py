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

    Mirrors datetime polars_lazy: no ``ReservedColumnsTestMixin`` here. There is
    no reserved-column guard on any backend; the SQLite test class carries the
    acceptance mixin for the cross-backend coverage.
    """

    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyTimeBucketization
