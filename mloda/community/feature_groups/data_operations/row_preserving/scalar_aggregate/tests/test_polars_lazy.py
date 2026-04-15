"""Tests for PolarsLazyScalarAggregate compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("polars")

from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.polars_lazy_scalar_aggregate import (
    PolarsLazyScalarAggregate,
)
from mloda.testing.feature_groups.data_operations.row_preserving.scalar_aggregate.scalar_aggregate import (
    ScalarAggregateTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.polars_lazy import PolarsLazyTestMixin


class TestPolarsLazyScalarAggregate(PolarsLazyTestMixin, ScalarAggregateTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyScalarAggregate

    def test_collision_masked_src(self) -> None:
        """User column named __mloda_masked_src__ must survive the mask-enabled path."""
        self._run_collision_case("__mloda_masked_src__", use_mask=True)
