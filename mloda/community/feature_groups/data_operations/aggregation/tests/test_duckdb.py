"""Tests for DuckdbAggregation compute implementation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pytest

duckdb = pytest.importorskip("duckdb")

from mloda.community.feature_groups.data_operations.aggregation.duckdb_aggregation import (
    DuckdbAggregation,
)
from mloda.testing.feature_groups.data_operations.aggregation.aggregation import (
    AggregationTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.frameworks import DuckdbTestMixin, duckdb_below_2


class TestDuckdbAggregation(DuckdbTestMixin, AggregationTestBase):
    """Standard tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbAggregation

    @classmethod
    def pinned_divergences(cls) -> Mapping[str, Mapping[str, list[Any]]]:
        pins: dict[str, Mapping[str, list[Any]]] = {
            **super().pinned_divergences(),
            # The NaN partition is returned after ordinary keys instead of in first-occurrence order.
            "8-partition_keys-val__sum_agg": {"grp": [1.0, 2.0, math.nan], "val__sum_agg": [100.0, 200.0, 30.0]},
        }
        if duckdb_below_2():
            # DuckDB 1.x loses the negative sign of the first-seen zero partition key.
            pins["6-signed_zero-v__count_agg"] = {"k": [0.0], "v__count_agg": [3]}
        return pins
