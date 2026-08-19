"""Tests for PythonDictTimeBucketization compute implementation."""

from __future__ import annotations

from typing import Any

import pytest

from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.python_dict_time_bucketization import (  # noqa: E501
    PythonDictTimeBucketization,
)
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.time_bucketization.time_bucketization import (
    TimeBucketizationTestBase,
)


class TestPythonDictTimeBucketization(PythonDictTestMixin, TimeBucketizationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictTimeBucketization

    # -- Non-divisor bucket sizes (epoch-anchored floor regression guards) --

    def test_cross_framework_non_divisor_minute_bucket(self) -> None:
        """``n=7`` does not evenly divide 60: ``_floor_dt``'s ``(minute // n) * n``
        anchoring (reset to zero at the top of the enclosing hour) only agrees
        with PyArrow's epoch-anchored ``floor_temporal`` (multiples of the
        bucket duration since 1970-01-01 UTC) when ``n`` divides 60 evenly.
        Must match the live PyArrow oracle row-for-row via the shared
        ``_compare_bucket_with_reference`` mechanism (no hand-computed values).
        """
        self._compare_bucket_with_reference("timestamp__floor_7_minute")

    def test_cross_framework_non_divisor_hour_bucket(self) -> None:
        """``n=5`` does not evenly divide 24: same enclosing-day-vs-epoch
        anchoring mismatch as the 7-minute case, one level up. Must match the
        live PyArrow oracle row-for-row via the shared
        ``_compare_bucket_with_reference`` mechanism (no hand-computed values).
        """
        self._compare_bucket_with_reference("timestamp__floor_5_hour")


class TestPythonDictDateSourceRejected:
    """DATE-only source columns (Python datetime.date, no time component) are rejected.

    A pa.date32() Arrow column round-trips to plain datetime.date objects,
    which lack hour/minute/second and cannot be bucketized. Mirrors the
    DuckDB precedent (test_duckdb.TestDuckdbDateSourceRejected): the
    source-type guard rejects DATE-only columns with a clear ValueError
    before any bucket math runs.
    """

    def test_date_column_rejected(self) -> None:
        from datetime import date

        from mloda.testing.feature_groups.data_operations.helpers import make_feature_set

        data = {"timestamp": [date(2023, 1, 1)]}
        fs = make_feature_set("timestamp__floor_1_day")
        with pytest.raises(ValueError, match=r"(?i)timestamp|datetime|DATE"):
            PythonDictTimeBucketization.calculate_feature(data, fs)


class TestPythonDictNonUtcTimezoneSupported:
    """Non-UTC tz-aware sources are fully supported, unlike SQLite's TEXT-storage guard.

    pa.Table.to_pylist() converts tz-aware Arrow timestamps to Python
    datetime objects carrying real zoneinfo.ZoneInfo tzinfo, so DST-aware
    offsets recompute automatically on replace()/timedelta arithmetic and no
    rejection guard is needed here. Counterpart to
    test_sqlite_result_type.TestSqliteResultTypeContract.test_dst_zone_month_floor_rejected:
    identical DST-crossing input, opposite contract.
    """

    def test_dst_zone_month_floor_matches_pyarrow_oracle(self) -> None:
        from datetime import datetime

        import pyarrow as pa

        from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.pyarrow_time_bucketization import (  # noqa: E501
            PyArrowTimeBucketization,
        )
        from mloda.testing.feature_groups.data_operations.helpers import make_feature_set

        arrow_table = pa.table(
            {
                "timestamp": pa.array(
                    [datetime(2023, 3, 31, 12, 0, 0)],
                    type=pa.timestamp("us", tz="Europe/Berlin"),
                ),
            }
        )
        data = {name: arrow_table.column(name).to_pylist() for name in arrow_table.column_names}
        fs = make_feature_set("timestamp__floor_1_month")

        result = PythonDictTimeBucketization.calculate_feature(data, fs)
        oracle = PyArrowTimeBucketization.calculate_feature(arrow_table, fs)

        actual = result["timestamp__floor_1_month"][0]
        expected = oracle.column("timestamp__floor_1_month").to_pylist()[0]
        assert actual == expected, f"{actual!r} != oracle {expected!r}"
        assert actual.utcoffset() == expected.utcoffset(), (
            f"expected matching UTC offset (DST-correct): {actual.utcoffset()!r} != {expected.utcoffset()!r}"
        )

    def test_dst_zone_month_floor_recomputes_offset_for_pytz_tzinfo(self) -> None:
        """A ``pytz`` ``DstTzInfo`` is pinned at construction (offset fixed to
        whatever transition period the original ``localize()`` call resolved)
        and does NOT recompute the DST offset when ``.replace(...)``-ed onto a
        different date -- unlike ``zoneinfo.ZoneInfo``, whose ``utcoffset()``
        resolves dynamically per-datetime. ``test_dst_zone_month_floor_matches_pyarrow_oracle``
        above builds its input via ``pa.array(...).to_pylist()``, whose choice of
        tzinfo *type* is pyarrow/pandas-version-dependent: on the CI Python 3.10
        runner that path happens to attach a pytz ``DstTzInfo``, while 3.11+ in
        the same CI matrix get ``zoneinfo.ZoneInfo`` -- so that test passes here
        (this environment's pyarrow attaches ``zoneinfo.ZoneInfo``) but fails on
        the 3.10 runner. This test instead constructs the pytz ``DstTzInfo``
        directly, so the bug reproduces deterministically regardless of which
        tzinfo type the locally installed pyarrow/pandas combination attaches.
        """
        from datetime import datetime, timedelta

        from mloda.testing.feature_groups.data_operations.helpers import make_feature_set

        pytz = pytest.importorskip("pytz")

        berlin = pytz.timezone("Europe/Berlin")
        # DST (CEST, +2:00) is in effect: Europe/Berlin's 2023 DST window runs
        # March 26 - October 29.
        dt = berlin.localize(datetime(2023, 3, 31, 12, 0, 0))
        data = {"timestamp": [dt]}
        fs = make_feature_set("timestamp__floor_1_month")

        result = PythonDictTimeBucketization.calculate_feature(data, fs)

        actual = result["timestamp__floor_1_month"][0]
        assert (actual.year, actual.month, actual.day, actual.hour, actual.minute, actual.second) == (
            2023,
            3,
            1,
            0,
            0,
            0,
        ), f"expected floor to 2023-03-01 00:00; got {actual!r}"
        assert actual.utcoffset() == timedelta(hours=1), (
            "March 1 2023 in Europe/Berlin is CET (+1:00); DST (CEST, +2:00) does not start "
            f"until March 26. Got utcoffset {actual.utcoffset()!r} -- the original March 31 "
            "CEST offset was carried over onto the floored March 1 date instead of being "
            "re-resolved for it."
        )


class TestPythonDictRoundDstCrossing:
    """``round``'s distance-to-boundary math must use absolute (not wall-clock) elapsed time.

    ``_round_dt`` computes ``offset = dt - floored`` and ``length = next_boundary
    - floored`` from ``_floor_dt`` / ``_next_boundary`` results that carry a
    genuine ``zoneinfo.ZoneInfo`` tzinfo, so the ``datetime`` subtraction itself
    is UTC-instant-based -- but ``_floor_dt``'s ``day`` floor and
    ``_next_boundary``'s ``+timedelta(days=n)`` derive the boundary
    datetimes from wall-clock replace()/arithmetic, which silently assumes
    every local day is exactly 24 wall-clock hours. On a DST transition day
    that assumption is false (a spring-forward day is only 23 real hours), so
    the computed ``length`` and thus the midpoint are wrong relative to
    PyArrow's DST-aware ``round_temporal``. Compared against the live PyArrow
    oracle, mirroring TestPythonDictNonUtcTimezoneSupported above.
    """

    def test_round_1_day_spring_forward_matches_pyarrow_oracle(self) -> None:
        from datetime import datetime

        import pyarrow as pa

        from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.pyarrow_time_bucketization import (  # noqa: E501
            PyArrowTimeBucketization,
        )
        from mloda.testing.feature_groups.data_operations.helpers import make_feature_set

        # pa.array() interprets a naive Python datetime as a UTC instant, so
        # 16:00 UTC on 2023-03-12 displays as 12:00:00-04:00 (EDT) in
        # America/New_York -- local noon on the spring-forward day, where
        # clocks jump 02:00 -> 03:00 (the day is 23 real hours, not 24).
        arrow_table = pa.table(
            {
                "timestamp": pa.array(
                    [datetime(2023, 3, 12, 16, 0, 0)],
                    type=pa.timestamp("us", tz="America/New_York"),
                ),
            }
        )
        data = {name: arrow_table.column(name).to_pylist() for name in arrow_table.column_names}
        fs = make_feature_set("timestamp__round_1_day")

        result = PythonDictTimeBucketization.calculate_feature(data, fs)
        oracle = PyArrowTimeBucketization.calculate_feature(arrow_table, fs)

        actual = result["timestamp__round_1_day"][0]
        expected = oracle.column("timestamp__round_1_day").to_pylist()[0]
        assert actual == expected, f"{actual!r} != oracle {expected!r}"

    def test_round_1_day_spring_forward_recomputes_offset_for_pytz_tzinfo(self) -> None:
        """Companion to ``test_round_1_day_spring_forward_matches_pyarrow_oracle``,
        constructing the pytz ``DstTzInfo`` directly instead of going through
        ``pa.array(...).to_pylist()`` (whose resulting tzinfo *type* is
        pyarrow/pandas-version-dependent -- see the sibling test in
        ``TestPythonDictNonUtcTimezoneSupported`` above for the same
        environment-dependence). This reproduces the bug deterministically
        regardless of which tzinfo type the locally installed pyarrow/pandas
        combination attaches.

        With a real (dynamic) ``zoneinfo.ZoneInfo`` tzinfo, this rounds DOWN
        to 2023-03-12 00:00 EST: the spring-forward day is 23 real hours (the
        02:00 -> 03:00 jump), so wall-clock noon is only 11 real hours past
        midnight -- short of the 11.5-hour midpoint of a 23-real-hour day.
        With pytz's ``DstTzInfo`` pinned to EDT (-04:00, the offset baked in
        by the original noon ``localize()`` call), ``_floor_dt``'s
        ``.replace(tzinfo=...)`` carries that EDT offset onto the March-12
        *and* March-13 boundaries alike, making the day look like a plain 24
        real hours (noon is then exactly at, i.e. past-or-at, the midpoint)
        -- rounding UP to 2023-03-13 00:00 instead.
        """
        from datetime import datetime

        from mloda.testing.feature_groups.data_operations.helpers import make_feature_set

        pytz = pytest.importorskip("pytz")

        eastern = pytz.timezone("America/New_York")
        # Local noon, already past the 02:00 -> 03:00 spring-forward transition.
        dt = eastern.localize(datetime(2023, 3, 12, 12, 0, 0))
        data = {"timestamp": [dt]}
        fs = make_feature_set("timestamp__round_1_day")

        result = PythonDictTimeBucketization.calculate_feature(data, fs)

        actual = result["timestamp__round_1_day"][0]
        assert (actual.year, actual.month, actual.day, actual.hour, actual.minute, actual.second) == (
            2023,
            3,
            12,
            0,
            0,
            0,
        ), (
            "expected floor to 2023-03-12 00:00 (11 real hours from midnight to noon is short "
            f"of the 23-real-hour day's 11.5-hour midpoint); got {actual!r}"
        )
