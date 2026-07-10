"""Tests for SqliteScalarAggregate compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.options import Options
from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.sqlite_scalar_aggregate import (
    SqliteScalarAggregate,
)
from mloda.testing.feature_groups.data_operations.row_preserving.scalar_aggregate.scalar_aggregate import (
    ScalarAggregateTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.sqlite import SqliteTestMixin


class TestSqliteScalarAggregate(CapabilityHookTestMixin, SqliteTestMixin, ScalarAggregateTestBase):
    @classmethod
    def supported_agg_types(cls) -> set[str]:
        return {"sum", "min", "max", "avg", "mean", "count"}

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteScalarAggregate

    @classmethod
    def capability_supported(cls) -> tuple[tuple[str, Options], ...]:
        return (("value__sum_scalar", Options()),)

    @classmethod
    def capability_unsupported(cls) -> tuple[tuple[str, Options], ...]:
        return (("value__median_scalar", Options()),)
