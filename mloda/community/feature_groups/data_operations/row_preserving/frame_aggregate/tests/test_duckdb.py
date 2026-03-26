"""Tests for DuckDB frame aggregate implementation."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

duckdb = pytest.importorskip("duckdb")

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.user import Feature
from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.duckdb_frame_aggregate import (
    DuckDBFrameAggregate,
)


@pytest.fixture()
def duckdb_data() -> DuckdbRelation:
    conn = duckdb.connect(":memory:")
    arrow_table = pa.table({
        "region": ["A", "A", "A", "A", "B", "B", "B"],
        "timestamp": [1, 2, 3, 4, 1, 2, 3],
        "value": [10, 20, 30, 40, 100, 200, 300],
    })
    rel = conn.from_arrow(arrow_table)
    return DuckdbRelation(conn, rel)


class TestDuckDBRolling:
    def test_rolling_sum_2(self, duckdb_data: DuckdbRelation) -> None:
        feature = Feature(
            "value__sum_rolling_2",
            options=Options(context={"partition_by": ["region"], "order_by": "timestamp"}),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = DuckDBFrameAggregate.calculate_feature(duckdb_data, fs)
        col = result.to_arrow_table().column("value__sum_rolling_2").to_pylist()

        assert col[0] == 10
        assert col[1] == 30
        assert col[2] == 50
        assert col[3] == 70


class TestDuckDBCumulative:
    def test_cumsum(self, duckdb_data: DuckdbRelation) -> None:
        feature = Feature(
            "value__cumsum",
            options=Options(context={"partition_by": ["region"], "order_by": "timestamp"}),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = DuckDBFrameAggregate.calculate_feature(duckdb_data, fs)
        col = result.to_arrow_table().column("value__cumsum").to_pylist()

        assert col[0] == 10
        assert col[1] == 30
        assert col[2] == 60
        assert col[3] == 100
