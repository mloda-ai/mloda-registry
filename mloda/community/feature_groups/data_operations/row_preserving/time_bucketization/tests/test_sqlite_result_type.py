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

    def test_non_utc_tz_aware_rejected(self) -> None:
        """Non-UTC tz-aware sources must be rejected up-front.

        SQLite stores tz-aware Arrow timestamps as TEXT with only the
        numeric ``+HH:MM`` offset, losing the IANA zone. Wall-clock floor
        with the source row's offset reattached silently mis-computes
        across DST transitions (e.g. ``Europe/Berlin 2023-03-31 14:00+02:00``
        month-floor should yield ``2023-03-01 00:00+01:00`` per PyArrow,
        but a stored-offset strategy would yield ``+02:00``). Refuse the
        input instead of returning a subtly wrong result.
        """
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
        with pytest.raises(ValueError, match=r"(?i)non-utc|timezone|tz"):
            SqliteTimeBucketization().calculate_feature(rel, fs)

    def test_dst_zone_month_floor_rejected(self) -> None:
        """Direct regression for the DST-crossing month-floor bug.

        ``Europe/Berlin 2023-03-31 12:00:00`` is stored as ``+02:00`` (CEST).
        PyArrow floors the month to ``2023-03-01 00:00:00+01:00`` (CET).
        A wall-clock floor with the source offset reattached would emit
        ``+02:00`` -- a 1-hour error in UTC instant. Rejection at validation
        is the correct response under SQLite's TEXT-only storage.
        """
        tbl = pa.table(
            {
                "timestamp": pa.array(
                    [datetime(2023, 3, 31, 12, 0, 0)],
                    type=pa.timestamp("us", tz="Europe/Berlin"),
                ),
            }
        )
        rel = SqliteRelation.from_arrow(sqlite3.connect(":memory:"), tbl)
        fs = make_feature_set("timestamp__floor_1_month")
        with pytest.raises(ValueError, match=r"(?i)non-utc|timezone|tz"):
            SqliteTimeBucketization().calculate_feature(rel, fs)

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

    def test_malformed_later_row_rejected(self) -> None:
        """A TEXT column whose first non-null row parses but a later row doesn't must be rejected.

        SQLite's ``julianday`` returns NULL (rather than raising) for
        unparsable input, so a LIMIT 1 probe would let the malformed row
        slip past validation and silently emit NULL at compute time. The
        validator must scan the whole column.
        """
        con = sqlite3.connect(":memory:")
        # Build a relation via from_arrow so the rest of the contract is intact,
        # then sneak in a malformed row by direct INSERT.
        tbl = pa.table({"timestamp": pa.array(["2023-01-01 00:00:00"], type=pa.string())})
        rel = SqliteRelation.from_arrow(con, tbl)
        con.execute(f'INSERT INTO "{rel.table_name}" VALUES (?)', ("not-a-date",))  # nosec
        fs = make_feature_set("timestamp__floor_1_day")
        with pytest.raises(ValueError, match=r"(?i)timestamp|datetime|parse"):
            SqliteTimeBucketization().calculate_feature(rel, fs)

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
