"""Tests for PolarsLazyPercentile compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("polars")

from mloda.community.feature_groups.data_operations.row_preserving.percentile.polars_lazy_percentile import (
    PolarsLazyPercentile,
)
from mloda.testing.feature_groups.data_operations.row_preserving.percentile.percentile import (
    PercentileTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.polars_lazy import PolarsLazyTestMixin


class TestPolarsLazyPercentile(PolarsLazyTestMixin, PercentileTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyPercentile

    def test_collision_masked_src(self) -> None:
        """User column named __mloda_masked_src__ must survive the mask-enabled path."""
        self._run_collision_case("__mloda_masked_src__", use_mask=True)
