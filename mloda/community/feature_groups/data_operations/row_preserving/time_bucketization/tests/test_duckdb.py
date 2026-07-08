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
    """DATE source columns are rejected up-front with a clear ValueError.

    DuckDB has a native DATE type, but it is intentionally absent from
    _DUCKDB_TIMESTAMP_PREFIXES, so _assert_source_column_is_timestamp
    rejects a DATE column before any SQL runs. This avoids the cryptic
    BinderException about epoch(BIGINT) that DuckDB would otherwise raise
    when rounding a DATE at sub-day units. Users with DATE columns should
    cast to TIMESTAMP before bucketing.
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


class TestDuckdbTimeBucketizationCalendarRound:
    """Calendar-unit ROUND (week / month / year), which the shared base does not cover.

    Guards the calendar branch of ``_round_expression`` against the PyArrow oracle.
    """

    @pytest.mark.parametrize(
        "feature_name",
        [
            "timestamp__round_1_week",
            "timestamp__round_1_month",
            "timestamp__round_1_year",
        ],
    )
    def test_calendar_round_matches_pyarrow_oracle(self, feature_name: str) -> None:
        import duckdb
        from datetime import datetime
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
        from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
        from mloda.testing.feature_groups.data_operations.mixins.duckdb import pin_connection_utc_via_core
        from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.pyarrow_time_bucketization import (  # noqa: E501
            PyArrowTimeBucketization,
        )

        arrow_table = _create_bucket_arrow_table()

        con = duckdb.connect()
        pin_connection_utc_via_core(con)
        rel = DuckdbRelation.from_arrow(con, arrow_table)

        fs = make_feature_set(feature_name)
        result = DuckdbTimeBucketization.calculate_feature(rel, fs)
        oracle = PyArrowTimeBucketization.calculate_feature(arrow_table, fs)

        actual = list(result.to_arrow_table().column(feature_name).to_pylist())
        expected = list(oracle.column(feature_name).to_pylist())

        assert len(actual) == len(expected), f"row count {len(actual)} != oracle {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            if e is None:
                assert a is None, f"row {i}: expected None, got {a!r}"
            else:
                # Both are tz-aware UTC datetimes; ``==`` compares instants.
                if isinstance(a, str):
                    a = datetime.fromisoformat(a)
                assert a == e, f"row {i}: {a!r} != oracle {e!r}"


class TestDuckdbTimeBucketizationNonUtcSession:
    """Flooring stays UTC-anchored on a non-UTC session, via the core chokepoint.

    Presets a non-UTC session, then relies on the core UTC guarantee (not a
    per-FG pin) to anchor the result.
    """

    def test_floor_1_day_utc_anchored_on_non_utc_session(self) -> None:
        import duckdb
        from datetime import datetime
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
        from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
        from mloda.testing.feature_groups.data_operations.mixins.duckdb import pin_connection_utc_via_core

        con = duckdb.connect()
        con.execute("SET TimeZone='America/New_York'")
        pin_connection_utc_via_core(con)

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
