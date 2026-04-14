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
from mloda.testing.feature_groups.data_operations.mixins.reserved_columns import ReservedColumnsTestMixin


class TestPolarsLazyScalarAggregate(ReservedColumnsTestMixin, PolarsLazyTestMixin, ScalarAggregateTestBase):
    @classmethod
    def implementation_class(cls) -> Any:
        return PolarsLazyScalarAggregate

    @classmethod
    def reserved_columns_feature_name(cls) -> str:
        return "value_int__sum_scalar"

    @classmethod
    def reserved_columns_partition_by(cls) -> list[str] | None:
        return None

    @classmethod
    def reserved_columns_order_by(cls) -> str | None:
        return None
