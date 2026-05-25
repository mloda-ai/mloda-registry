"""Tests for PandasTimeBucketization compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pandas")

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
