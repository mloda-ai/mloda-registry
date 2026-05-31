"""Tests for DuckdbTimeBucketization compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("duckdb")

from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.duckdb_time_bucketization import (
    DuckdbTimeBucketization,
)
from mloda.testing.feature_groups.data_operations.mixins.duckdb import DuckdbTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.time_bucketization.time_bucketization import (
    EXPECTED_FLOOR_1_DAY,
    TimeBucketizationTestBase,
    _create_bucket_arrow_table,
)


class TestDuckdbTimeBucketization(DuckdbTestMixin, TimeBucketizationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbTimeBucketization


class TestDuckdbDateSourceRejected:
    """DATE source columns are rejected with a clear ValueError.

    DuckDB has a native DATE type. The current code accepts DATE because of
    the 'DATE' entry in _DUCKDB_TIMESTAMP_PREFIXES, but round at sub-day
    units fails inside the SQL with a cryptic BinderException about
    epoch(BIGINT). Reject up-front instead and leave timestamp coercion
    to the caller.
    """

    def test_date_column_rejected(self) -> None:
        import duckdb
        import pyarrow as pa
        from datetime import date
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
        from mloda.testing.feature_groups.data_operations.helpers import make_feature_set

        con = duckdb.connect(":memory:")
        date_table = pa.table({"timestamp": pa.array([date(2023, 1, 1)], type=pa.date32())})
        rel = DuckdbRelation.from_arrow(con, date_table)
        fs = make_feature_set("timestamp__floor_1_day")
        with pytest.raises(ValueError, match=r"(?i)timestamp|datetime|DATE"):
            DuckdbTimeBucketization.calculate_feature(rel, fs)


class TestDuckdbTimeBucketizationNonUtcSession:
    """Bucketization must produce UTC-anchored instants regardless of session TZ (issue #238).

    DuckDB's ``DATE_TRUNC`` / ``time_bucket`` operate in the connection's
    session timezone. On a non-UTC session, a UTC input such as
    ``2023-01-01 00:00 UTC`` floors to *local* midnight (a different instant)
    instead of UTC midnight. The feature group must pin the session timezone
    to UTC inside ``_compute_bucket`` so the result is correct no matter what
    the session timezone was preset to.

    This test builds its own connection (rather than the mixin's) so it can
    deterministically reproduce the bug by setting a non-UTC session timezone,
    independent of the host OS timezone.
    """

    def test_floor_1_day_utc_anchored_on_non_utc_session(self) -> None:
        import duckdb
        from datetime import datetime
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
        from mloda.testing.feature_groups.data_operations.helpers import make_feature_set

        con = duckdb.connect()
        # Preset a NON-UTC session timezone; this is what triggers the bug.
        con.execute("SET TimeZone='America/New_York'")

        rel = DuckdbRelation.from_arrow(con, _create_bucket_arrow_table())
        fs = make_feature_set("timestamp__floor_1_day")
        result = DuckdbTimeBucketization.calculate_feature(rel, fs)

        col = list(result.to_arrow_table().column("timestamp__floor_1_day").to_pylist())

        assert len(col) == len(EXPECTED_FLOOR_1_DAY), f"row count {len(col)} != expected {len(EXPECTED_FLOOR_1_DAY)}"
        for i, (actual, expected) in enumerate(zip(col, EXPECTED_FLOOR_1_DAY)):
            if expected is None:
                assert actual is None, f"row {i}: expected None, got {actual!r}"
            else:
                # Both are tz-aware datetimes; ``==`` compares instants.
                if isinstance(actual, str):
                    actual = datetime.fromisoformat(actual)
                assert actual == expected, f"row {i}: {actual!r} != {expected!r}"
