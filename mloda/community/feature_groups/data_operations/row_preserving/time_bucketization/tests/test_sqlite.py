"""Tests for SqliteTimeBucketization compute implementation."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import date
from typing import Any

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa
from mloda_plugins.compute_framework.base_implementations.sqlite.sqlite_relation import SqliteRelation

from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.sqlite_time_bucketization import (
    SqliteTimeBucketization,
)
from mloda.testing.feature_groups.data_operations.helpers import calculate_with, make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.frameworks import (
    SqliteComputedColumnTestMixin,
    SqliteTestMixin,
    SqliteViewTestMixin,
)
from mloda.testing.feature_groups.data_operations.mixins.reserved_columns import ReservedColumnsTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.time_bucketization.time_bucketization import (
    TimeBucketizationTestBase,
)


class TestSqliteTimeBucketization(ReservedColumnsTestMixin, SqliteTestMixin, TimeBucketizationTestBase):
    """All tests inherited from the base class.

    Uses ``ReservedColumnsTestMixin`` to assert that a ``__mloda_``-prefixed
    user column is accepted: there is no reserved-column guard any more, and
    SQLite time bucketization keeps input order without adding a helper
    column, so such inputs are processed normally.
    """

    @classmethod
    def implementation_class(cls) -> Any:
        return SqliteTimeBucketization

    @classmethod
    def rejects_non_utc_source(cls) -> bool:
        return True

    @classmethod
    def reserved_columns_feature_name(cls) -> str:
        return "timestamp__floor_1_day"

    @classmethod
    def reserved_columns_partition_by(cls) -> list[str] | None:
        return None


class TestSqliteTimeBucketizationViewInput(SqliteViewTestMixin, TestSqliteTimeBucketization):
    """Runs the TimeBucketization suite on a view input."""


class TestSqliteTimeBucketizationComputedInput(SqliteComputedColumnTestMixin, TestSqliteTimeBucketization):
    """Runs the TimeBucketization suite on computed (empty declared type) input columns."""


def _date_affinity_source() -> SqliteRelation:
    con = sqlite3.connect(":memory:")
    con.execute('CREATE TABLE t ("timestamp" DATE)')
    con.execute("INSERT INTO t VALUES ('2024-01-01')")
    return SqliteRelation(con, "t")


def _date32_source() -> SqliteRelation:
    tbl = pa.table({"timestamp": pa.array([date(2024, 1, 1)], type=pa.date32())})
    return SqliteRelation.from_arrow(sqlite3.connect(":memory:"), tbl)


def _string_source(*values: str) -> Callable[[], SqliteRelation]:
    def build() -> SqliteRelation:
        tbl = pa.table({"timestamp": pa.array(list(values), type=pa.string())})
        return SqliteRelation.from_arrow(sqlite3.connect(":memory:"), tbl)

    return build


def _blob_source() -> SqliteRelation:
    tbl = pa.table({"timestamp": pa.array([b"2024-01-01 14:00:00+01:00"], type=pa.binary())})
    return SqliteRelation.from_arrow(sqlite3.connect(":memory:"), tbl)


def _julian_integer_source() -> SqliteRelation:
    con = sqlite3.connect(":memory:")
    con.execute('CREATE TABLE t ("timestamp" DATETIME)')
    con.execute("INSERT INTO t VALUES (2460000)")
    return SqliteRelation(con, "t")


class TestSqliteSourceShapes:
    """Source values without a time of day or in a non-text storage class are rejected."""

    @pytest.mark.parametrize(
        "build_source, match",
        [
            pytest.param(_string_source("2024-01-01"), r"value '2024-01-01'", id="date-text"),
            pytest.param(_date_affinity_source, r"value '2024-01-01'", id="date-affinity"),
            pytest.param(_date32_source, r"value '2024-01-01'", id="date32"),
            pytest.param(
                _string_source("2024-01-01T14:00:00", "2024-01-01"), r"value '2024-01-01'", id="date-later-row"
            ),
            pytest.param(_string_source("14:00"), r"value '14:00'", id="time-only"),
            pytest.param(_string_source("now"), r"value 'now'", id="now"),
            pytest.param(
                _string_source("2024-01-01 14:00:00+01:00 "),
                r"value '2024-01-01 14:00:00\+01:00 '",
                id="trailing-space",
            ),
            pytest.param(_blob_source, r"storage class 'blob'", id="blob"),
            pytest.param(_julian_integer_source, r"storage class 'integer'", id="julian-integer"),
        ],
    )
    def test_non_timestamp_value_rejected(self, build_source: Callable[[], SqliteRelation], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            calculate_with(SqliteTimeBucketization, build_source(), make_feature_set("timestamp__floor_1_day"))

    @pytest.mark.parametrize(
        "value, feature, expected",
        [
            ("2024-01-01T14:00:00Z", "timestamp__floor_1_day", "2024-01-01 00:00:00+00:00"),
            ("2024-01-01T14:00:00z", "timestamp__floor_1_day", "2024-01-01 00:00:00+00:00"),
            ("2024-01-01T14:00:00Z", "timestamp__floor_1_month", "2024-01-01 00:00:00+00:00"),
            ("2024-01-01T14:00:00.5Z", "timestamp__ceil_1_minute", "2024-01-01 14:01:00+00:00"),
            ("2024-01-01 14:00", "timestamp__floor_1_hour", "2024-01-01 14:00:00"),
        ],
    )
    def test_accepted_value_bucketed(self, value: str, feature: str, expected: str) -> None:
        result = calculate_with(SqliteTimeBucketization, _string_source(value)(), make_feature_set(feature))
        assert result.to_arrow_table().column(feature).to_pylist() == [expected]
