"""Tests for SqliteAggregation compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.core.abstract_plugins.components.options import Options
from mloda.community.feature_groups.data_operations.aggregation.sqlite_aggregation import (
    SqliteAggregation,
)
from mloda.testing.feature_groups.data_operations.aggregation.aggregation import (
    AggregationTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.sqlite import SqliteTestMixin


class TestSqliteAggregation(CapabilityHookTestMixin, SqliteTestMixin, AggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def supported_agg_types(cls) -> set[str]:
        return {"sum", "avg", "mean", "count", "min", "max"}

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteAggregation

    @classmethod
    def capability_supported(cls) -> tuple[tuple[str, Options], ...]:
        return (
            ("value__sum_agg", Options()),
            ("value__mean_agg", Options()),
        )

    @classmethod
    def capability_unsupported(cls) -> tuple[tuple[str, Options], ...]:
        return (
            ("value__median_agg", Options()),
            (
                "median_result",
                Options(
                    context={
                        "aggregation_type": "median",
                        "in_features": "value",
                        "partition_by": ["region"],
                    }
                ),
            ),
        )
