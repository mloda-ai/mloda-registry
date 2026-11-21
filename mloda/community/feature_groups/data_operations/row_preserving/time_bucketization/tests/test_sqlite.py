"""Tests for SqliteTimeBucketization compute implementation."""

from __future__ import annotations

import sqlite3
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


def _text_date_source() -> SqliteRelation:
    tbl = pa.table({"timestamp": pa.array(["2024-01-01"], type=pa.string())})
    return SqliteRelation.from_arrow(sqlite3.connect(":memory:"), tbl)


def _date_affinity_source() -> SqliteRelation:
    con = sqlite3.connect(":memory:")
    con.execute('CREATE TABLE t ("timestamp" DATE)')
    con.execute("INSERT INTO t VALUES ('2024-01-01')")
    return SqliteRelation(con, "t")


class TestSqliteDateOnlySource:
    """A date-only source value is bucketed as midnight, not mistaken for a tz offset."""

    @pytest.mark.parametrize("build_source", [_text_date_source, _date_affinity_source])
    @pytest.mark.parametrize(
        "feature, expected",
        [
            ("timestamp__floor_1_day", "2024-01-01 00:00:00"),
            ("timestamp__ceil_1_day", "2024-01-01 00:00:00"),
            ("timestamp__round_1_hour", "2024-01-01 00:00:00"),
            ("timestamp__ceil_1_week", "2024-01-08 00:00:00"),
        ],
    )
    def test_date_only_value_bucketed_as_midnight(self, build_source: Any, feature: str, expected: str) -> None:
        result = calculate_with(SqliteTimeBucketization, build_source(), make_feature_set(feature))
        assert result.to_arrow_table().column(feature).to_pylist() == [expected]

    def test_date_only_mixed_with_non_utc_offset_rejected(self) -> None:
        tbl = pa.table({"timestamp": pa.array(["2024-01-01", "2023-01-01 14:00:00+01:00"], type=pa.string())})
        rel = SqliteRelation.from_arrow(sqlite3.connect(":memory:"), tbl)
        with pytest.raises(ValueError, match="non-UTC"):
            calculate_with(SqliteTimeBucketization, rel, make_feature_set("timestamp__floor_1_day"))
