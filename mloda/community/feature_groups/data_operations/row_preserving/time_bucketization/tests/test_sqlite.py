"""Tests for SqliteTimeBucketization compute implementation."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing
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


@pytest.fixture
def connection() -> Iterator[sqlite3.Connection]:
    with closing(sqlite3.connect(":memory:")) as con:
        yield con


def _timestamp_source(connection: sqlite3.Connection, values: list[Any], affinity: str = "TEXT") -> SqliteRelation:
    connection.execute(f'CREATE TABLE t ("timestamp" {affinity})')  # nosec B608
    connection.executemany("INSERT INTO t VALUES (?)", [(value,) for value in values])
    return SqliteRelation(connection, "t")


class TestSqliteTimestampStorageClasses:
    """Timestamp affinities still require every non-null value to have text storage."""

    @pytest.mark.parametrize(
        "affinity, value, storage_class",
        [
            ("TEXT", b"2024-01-01 14:00:00+01:00", "blob"),
            ("DATETIME", b"2024-01-01 14:00:00+01:00", "blob"),
            ("TIMESTAMP", b"2024-01-01 14:00:00+01:00", "blob"),
            ("DATE", b"2024-01-01 14:00:00+01:00", "blob"),
            ("DATETIME", 2460311, "integer"),
            ("TIMESTAMP", 2460311, "integer"),
            ("DATE", 2460311, "integer"),
            ("DATETIME", 2460311.5, "real"),
            ("TIMESTAMP", 2460311.5, "real"),
            ("DATE", 2460311.5, "real"),
        ],
    )
    @pytest.mark.parametrize("mixed", [False, True], ids=["only-offender", "late-offender"])
    def test_non_text_storage_rejected(
        self, connection: sqlite3.Connection, affinity: str, value: Any, storage_class: str, mixed: bool
    ) -> None:
        values = ["2024-01-01 14:00:00+00:00"] * 9 + [None, value] if mixed else [value]
        rel = _timestamp_source(connection, values, affinity)
        assert connection.execute('SELECT typeof("timestamp") FROM t ORDER BY rowid DESC LIMIT 1').fetchone() == (
            storage_class,
        )

        with pytest.raises(
            ValueError, match=rf"Source column 'timestamp'.*timestamp/datetime.*storage class '{storage_class}'"
        ):
            calculate_with(SqliteTimeBucketization, rel, make_feature_set("timestamp__floor_1_day"))


class TestSqliteDateOnlySource:
    """A date-only value is not a timestamp, including after Arrow-to-SQLite transfer."""

    @pytest.mark.parametrize("affinity", ["TEXT", "DATETIME", "TIMESTAMP", "DATE"])
    @pytest.mark.parametrize(
        "feature",
        [
            "timestamp__floor_1_day",
            "timestamp__ceil_1_day",
            "timestamp__round_1_hour",
            "timestamp__ceil_1_week",
        ],
    )
    def test_date_only_value_rejected(self, connection: sqlite3.Connection, affinity: str, feature: str) -> None:
        rel = _timestamp_source(connection, ["2024-01-01"], affinity)
        with pytest.raises(ValueError, match=r"Source column 'timestamp' must be a timestamp/datetime"):
            calculate_with(SqliteTimeBucketization, rel, make_feature_set(feature))

    @pytest.mark.parametrize("arrow_type", [pa.date32(), pa.date64()])
    def test_arrow_date_source_rejected(self, connection: sqlite3.Connection, arrow_type: pa.DataType) -> None:
        tbl = pa.table({"timestamp": pa.array([date(2024, 1, 1), None], type=arrow_type)})
        rel = SqliteRelation.from_arrow(connection, tbl)
        with pytest.raises(ValueError, match=r"Source column 'timestamp' must be a timestamp/datetime"):
            calculate_with(SqliteTimeBucketization, rel, make_feature_set("timestamp__floor_1_day"))

    @pytest.mark.parametrize("computed", [False, True], ids=["declared", "computed"])
    def test_later_date_only_value_rejected(self, connection: sqlite3.Connection, computed: bool) -> None:
        rel = _timestamp_source(connection, ["2024-01-01 14:00:00+00:00"] * 9 + [None, "2024-01-01"])
        if computed:
            rel = rel.select(_raw_sql='+"timestamp" AS "timestamp"')
        with pytest.raises(ValueError, match=r"Source column 'timestamp' must be a timestamp/datetime"):
            calculate_with(SqliteTimeBucketization, rel, make_feature_set("timestamp__floor_1_day"))

    def test_timestamp_mixed_with_non_utc_offset_rejected(self, connection: sqlite3.Connection) -> None:
        rel = _timestamp_source(connection, ["2024-01-01 00:00:00", "2023-01-01 14:00:00+01:00"])
        with pytest.raises(ValueError, match=r"non-UTC.*'2023-01-01 14:00:00\+01:00'"):
            calculate_with(SqliteTimeBucketization, rel, make_feature_set("timestamp__floor_1_day"))


class TestSqliteTimestampContent:
    """Timestamp text and empty or null-only columns stay valid."""

    @pytest.mark.parametrize("affinity", ["TEXT", "DATETIME", "TIMESTAMP", "DATE", ""])
    @pytest.mark.parametrize("values", [[], [None, None]], ids=["empty", "all-null"])
    def test_empty_or_all_null_column_accepted(
        self, connection: sqlite3.Connection, affinity: str, values: list[Any]
    ) -> None:
        rel = _timestamp_source(connection, values, affinity)
        feature = "timestamp__floor_1_day"
        result = calculate_with(SqliteTimeBucketization, rel, make_feature_set(feature))
        assert result.to_arrow_table().column(feature).to_pylist() == values

    @pytest.mark.parametrize("affinity", ["TEXT", "DATETIME", "TIMESTAMP", "DATE"])
    def test_full_timestamp_text_accepted(self, connection: sqlite3.Connection, affinity: str) -> None:
        rel = _timestamp_source(connection, ["2024-01-01 14:00:00", None, "2024-01-01 14:00:00+00:00"], affinity)
        feature = "timestamp__floor_1_day"
        result = calculate_with(SqliteTimeBucketization, rel, make_feature_set(feature))
        assert result.to_arrow_table().column(feature).to_pylist() == [
            "2024-01-01 00:00:00",
            None,
            "2024-01-01 00:00:00+00:00",
        ]


class TestSqliteUtcZSuffix:
    """A trailing Z yields the same UTC text as +00:00 across fixed and calendar units."""

    @pytest.mark.parametrize(
        "feature, expected",
        [
            ("timestamp__floor_5_minute", "2024-01-17 14:35:00"),
            ("timestamp__ceil_1_hour", "2024-01-17 15:00:00"),
            ("timestamp__round_1_day", "2024-01-18 00:00:00"),
            ("timestamp__floor_1_week", "2024-01-15 00:00:00"),
            ("timestamp__ceil_1_month", "2024-02-01 00:00:00"),
            ("timestamp__round_1_year", "2024-01-01 00:00:00"),
        ],
    )
    @pytest.mark.parametrize("computed", [False, True], ids=["declared", "computed"])
    def test_utc_z_suffix_preserved(
        self, connection: sqlite3.Connection, feature: str, expected: str, computed: bool
    ) -> None:
        timestamp = "2024-01-17T14:37:30.000001"
        rel = _timestamp_source(connection, [timestamp + "Z", timestamp + "+00:00", timestamp, None])
        if computed:
            rel = rel.select(_raw_sql='+"timestamp" AS "timestamp"')
        result = calculate_with(SqliteTimeBucketization, rel, make_feature_set(feature))
        assert type(result) is SqliteRelation
        column = result.to_arrow_table().column(feature)
        assert column.type == pa.string()
        assert column.to_pylist() == [expected + "+00:00", expected + "+00:00", expected, None]
