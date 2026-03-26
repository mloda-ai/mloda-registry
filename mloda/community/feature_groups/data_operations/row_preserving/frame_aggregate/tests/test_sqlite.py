"""Tests for SQLite frame aggregate implementation."""

from __future__ import annotations

import sqlite3
from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.user import Feature
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.sqlite_frame_aggregate import (
    SqliteFrameAggregate,
)


@pytest.fixture()
def sqlite_data() -> SqliteRelation:
    conn = sqlite3.connect(":memory:")
    table = pa.table({
        "region": ["A", "A", "A", "A", "B", "B", "B"],
        "timestamp": [1, 2, 3, 4, 1, 2, 3],
        "value": [10, 20, 30, 40, 100, 200, 300],
    })
    return SqliteRelation.from_arrow(conn, table)


class TestSqliteRolling:
    def test_rolling_sum_2(self, sqlite_data: SqliteRelation) -> None:
        feature = Feature(
            "value__sum_rolling_2",
            options=Options(context={"partition_by": ["region"], "order_by": "timestamp"}),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = SqliteFrameAggregate.calculate_feature(sqlite_data, fs)
        col = list(result.to_arrow_table().column("value__sum_rolling_2").to_pylist())

        assert col[0] == 10
        assert col[1] == 30
        assert col[2] == 50
        assert col[3] == 70
        assert col[4] == 100
        assert col[5] == 300
        assert col[6] == 500


class TestSqliteCumulative:
    def test_cumsum(self, sqlite_data: SqliteRelation) -> None:
        feature = Feature(
            "value__cumsum",
            options=Options(context={"partition_by": ["region"], "order_by": "timestamp"}),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = SqliteFrameAggregate.calculate_feature(sqlite_data, fs)
        col = list(result.to_arrow_table().column("value__cumsum").to_pylist())

        assert col[0] == 10
        assert col[1] == 30
        assert col[2] == 60
        assert col[3] == 100
        assert col[4] == 100
        assert col[5] == 300
        assert col[6] == 600


class TestSqliteRowPreserving:
    def test_output_rows_equal_input(self, sqlite_data: SqliteRelation) -> None:
        feature = Feature(
            "value__sum_rolling_2",
            options=Options(context={"partition_by": ["region"], "order_by": "timestamp"}),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = SqliteFrameAggregate.calculate_feature(sqlite_data, fs)
        assert len(result) == 7
