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
from mloda.testing.feature_groups.data_operations.mixins.reserved_columns import ReservedColumnsTestMixin


class TestPolarsLazyPercentile(ReservedColumnsTestMixin, PolarsLazyTestMixin, PercentileTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyPercentile

    @classmethod
    def reserved_columns_feature_name(cls) -> str:
        return "value_int__p50_percentile"

    @classmethod
    def reserved_columns_partition_by(cls) -> list[str] | None:
        return ["region"]

    @classmethod
    def reserved_columns_order_by(cls) -> str | None:
        return None
