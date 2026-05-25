"""SQLite result-type contract tests for time bucketization.

Pins the post-refactor contract: ``SqliteTimeBucketization._compute_bucket``
returns a plain ``SqliteRelation`` whose result column is ``pa.string()`` in
ISO 8601 form, with the source tz suffix preserved (bucket math performed
in source-local wall-clock).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa

from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.sqlite_time_bucketization import (
    SqliteTimeBucketization,
)
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation


class TestSqliteResultTypeContract:
    """Post-refactor contract: plain SqliteRelation, string result column, tz suffix preserved."""

    def test_result_column_is_string_not_timestamp(self) -> None:
        tbl = pa.table(
            {
                "timestamp": pa.array(
                    [datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)],
                    type=pa.timestamp("us", tz="UTC"),
                ),
            }
        )
        rel = SqliteRelation.from_arrow(sqlite3.connect(":memory:"), tbl)
        fs = make_feature_set("timestamp__floor_1_day")
        result = SqliteTimeBucketization().calculate_feature(rel, fs)
        arrow = result.to_arrow_table()
        assert arrow.schema.field("timestamp__floor_1_day").type == pa.string()

    def test_result_is_plain_sqlite_relation(self) -> None:
        tbl = pa.table(
            {
                "timestamp": pa.array(
                    [datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)],
                    type=pa.timestamp("us", tz="UTC"),
                ),
            }
        )
        rel = SqliteRelation.from_arrow(sqlite3.connect(":memory:"), tbl)
        fs = make_feature_set("timestamp__floor_1_day")
        result = SqliteTimeBucketization().calculate_feature(rel, fs)
        assert type(result).__name__ == "SqliteRelation"
        assert not hasattr(result, "timestamp_result_columns")

    def test_utc_tz_suffix_preserved(self) -> None:
        tbl = pa.table(
            {
                "timestamp": pa.array(
                    [datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)],
                    type=pa.timestamp("us", tz="UTC"),
                ),
            }
        )
        rel = SqliteRelation.from_arrow(sqlite3.connect(":memory:"), tbl)
        fs = make_feature_set("timestamp__floor_1_day")
        result = SqliteTimeBucketization().calculate_feature(rel, fs)
        values = result.to_arrow_table().column("timestamp__floor_1_day").to_pylist()
        assert values[0] == "2023-01-01 00:00:00+00:00"

    def test_non_utc_tz_suffix_preserved(self) -> None:
        tbl = pa.table(
            {
                "timestamp": pa.array(
                    [datetime(2023, 1, 1, 14, 37, 23, tzinfo=timezone(timedelta(hours=1)))],
                    type=pa.timestamp("us", tz="Europe/Berlin"),
                ),
            }
        )
        rel = SqliteRelation.from_arrow(sqlite3.connect(":memory:"), tbl)
        fs = make_feature_set("timestamp__floor_1_day")
        result = SqliteTimeBucketization().calculate_feature(rel, fs)
        values = result.to_arrow_table().column("timestamp__floor_1_day").to_pylist()
        assert values[0] == "2023-01-01 00:00:00+01:00"

    def test_tz_naive_source(self) -> None:
        tbl = pa.table(
            {
                "timestamp": pa.array(
                    [datetime(2023, 1, 1, 14, 37, 23)],
                    type=pa.timestamp("us"),
                ),
            }
        )
        rel = SqliteRelation.from_arrow(sqlite3.connect(":memory:"), tbl)
        fs = make_feature_set("timestamp__floor_1_day")
        result = SqliteTimeBucketization().calculate_feature(rel, fs)
        values = result.to_arrow_table().column("timestamp__floor_1_day").to_pylist()
        assert isinstance(values[0], str)
        # Naive source: no tz offset suffix after seconds; last char must be a digit.
        assert values[0][-1].isdigit(), f"expected digit at end (no tz suffix), got {values[0]!r}"

    def test_null_propagates(self) -> None:
        tbl = pa.table(
            {
                "timestamp": pa.array(
                    [None],
                    type=pa.timestamp("us", tz="UTC"),
                ),
            }
        )
        rel = SqliteRelation.from_arrow(sqlite3.connect(":memory:"), tbl)
        fs = make_feature_set("timestamp__floor_1_day")
        result = SqliteTimeBucketization().calculate_feature(rel, fs)
        values = result.to_arrow_table().column("timestamp__floor_1_day").to_pylist()
        assert values[0] is None
