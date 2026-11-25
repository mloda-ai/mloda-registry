"""Tests for SqliteAggregation compute implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pyarrow as pa
from mloda.user import DataType, Options

from mloda.community.feature_groups.data_operations.aggregation.sqlite_aggregation import (
    SqliteAggregation,
)
from mloda.testing.feature_groups.data_operations.aggregation.aggregation import (
    AggregationTestBase,
)
from mloda.testing.feature_groups.data_operations.helpers import calculate_with, make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.frameworks import SqliteTestMixin
from mloda.testing.typed_data import TypedTestData


class TestSqliteAggregation(CapabilityHookTestMixin, SqliteTestMixin, AggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteAggregation

    @classmethod
    def pinned_divergences(cls) -> Mapping[str, Mapping[str, list[Any]]]:
        return {
            **super().pinned_divergences(),
            # SQLite loses the negative sign of the first-seen zero partition key.
            "6-signed_zero-v__count_agg": {"k": [0.0], "v__count_agg": [3]},
            # sqlite3 stores the NaN partition key as NULL.
            "8-partition_keys-val__sum_agg": {"grp": [None, 1.0, 2.0], "val__sum_agg": [30.0, 100.0, 200.0]},
        }

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

    def test_float_source_keeps_double_on_zero_rows(self) -> None:
        """Aggregating a float64 source over zero rows keeps the REAL type."""
        table = TypedTestData({"region": (DataType.STRING, []), "value": (DataType.DOUBLE, [])})
        data = self.create_test_data(table)
        fs = make_feature_set("value__sum_agg", ["region"])
        result = calculate_with(self.implementation_class(), data, fs)
        assert result.types[result.columns.index("value__sum_agg")] == pa.float64()
