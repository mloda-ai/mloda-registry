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
    """Guards calendar-unit ROUND (week / month / year) on the DuckDB backend.

    The shared ``TimeBucketizationTestBase`` only exercises ROUND at fixed-freq
    units (minute / hour / day), which take the ``{"minute", "hour", "day"}``
    branch of ``DuckdbTimeBucketization._round_expression``. The calendar-unit
    branch (week / month / year, whose bucket length varies per row) is never
    driven by the base suite, so a regression there goes unnoticed.

    Concretely, the calendar branch must reference the local floored-SQL string
    when it falls back to the floor value. A rename of the module-level ``floor``
    helper left the branch interpolating the FUNCTION object instead of the SQL
    string, so a request like ``timestamp__round_1_month`` emits SQL containing
    ``<function floor_expr at 0x...>`` and DuckDB raises a ParserException. This
    class is the standing regression guard for that calendar-round branch.

    Each unit is compared against the PyArrow oracle
    (``PyArrowTimeBucketization``) on the same arrow fixture, mirroring how the
    resample suite compares DuckDB against ``PyArrowResample`` rather than
    hand-computing expected literals.
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
        # Route through the real core chokepoint (UTC guarantee), exactly as production does.
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
    """Bucketization stays UTC-anchored regardless of session TZ, via the core chokepoint.

    DuckDB's ``DATE_TRUNC`` / ``time_bucket`` operate in the connection's
    session timezone. On a non-UTC session, a UTC input such as
    ``2023-01-01 00:00 UTC`` would floor to local midnight (a different instant)
    instead of UTC midnight. The feature group no longer pins the session
    timezone itself; UTC anchoring now comes from the core framework chokepoint
    (``DuckDBFramework.set_framework_connection_object``, mloda 0.9.0), which
    pins the session to UTC on first connection assignment.

    This test builds its own connection (rather than the mixin's) so it can
    preset a non-UTC session timezone, then routes that connection through the
    core chokepoint (as production does) to reproduce the real guarantee. It is
    the standing regression guard that UTC anchoring holds via the core guarantee,
    independent of the host OS timezone.
    """

    def test_floor_1_day_utc_anchored_on_non_utc_session(self) -> None:
        import duckdb
        from datetime import datetime
        from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation
        from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
        from mloda.testing.feature_groups.data_operations.mixins.duckdb import pin_connection_utc_via_core

        con = duckdb.connect()
        # Preset a NON-UTC session timezone; without the core guarantee this would misfloor.
        con.execute("SET TimeZone='America/New_York'")
        # Route through the real core chokepoint, exactly as production does.
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
