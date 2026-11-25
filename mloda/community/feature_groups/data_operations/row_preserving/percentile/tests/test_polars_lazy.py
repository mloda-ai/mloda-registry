"""Tests for PolarsLazyPercentile compute implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

pytest.importorskip("polars")

from mloda.community.feature_groups.data_operations.row_preserving.percentile.polars_lazy_percentile import (
    PolarsLazyPercentile,
)
from mloda.testing.feature_groups.data_operations.mixins.frameworks import PolarsLazyTestMixin
from mloda.testing.feature_groups.data_operations.mixins.reserved_columns import ReservedColumnsTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.percentile.percentile import (
    PercentileTestBase,
)


class TestPolarsLazyPercentile(ReservedColumnsTestMixin, PolarsLazyTestMixin, PercentileTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyPercentile

    @classmethod
    def reserved_columns_feature_name(cls) -> str:
        return "value_int__p50_percentile"

    @classmethod
    def pinned_divergences(cls) -> Mapping[str, Mapping[str, list[Any]]]:
        return {
            **super().pinned_divergences(),
            # Polars linear interpolation over an infinite neighbour gives NaN, not the infinite endpoint.
            "21-endpoints-v__p25_percentile": {"v__p25_percentile": [float("nan")] * 3},
        }
