"""Tests for DuckdbTimeBucketization compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("duckdb")

from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.duckdb_time_bucketization import (
    DuckdbTimeBucketization,
)
from mloda.testing.feature_groups.data_operations.mixins.duckdb import DuckdbTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.time_bucketization.time_bucketization import (
    TimeBucketizationTestBase,
)


class TestDuckdbTimeBucketization(DuckdbTestMixin, TimeBucketizationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbTimeBucketization
