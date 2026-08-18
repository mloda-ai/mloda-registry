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
