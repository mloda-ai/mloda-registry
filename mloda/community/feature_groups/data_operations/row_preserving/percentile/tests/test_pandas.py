"""Tests for PandasPercentile compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("pandas")

from mloda.community.feature_groups.data_operations.row_preserving.percentile.pandas_percentile import (
    PandasPercentile,
)
from mloda.testing.feature_groups.data_operations.mixins.pandas import (
    PandasNullableIntTestMixin,
    PandasTestMixin,
)
from mloda.testing.feature_groups.data_operations.row_preserving.percentile.percentile import (
    PercentileTestBase,
)


class TestPandasPercentile(PandasTestMixin, PercentileTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PandasPercentile


class TestPandasPercentileNullableInt(PandasNullableIntTestMixin, TestPandasPercentile):
    """Runs the percentile suite on nullable pandas int columns."""
