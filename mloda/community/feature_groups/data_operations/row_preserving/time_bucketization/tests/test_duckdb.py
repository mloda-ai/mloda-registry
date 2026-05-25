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
    TimeBucketizationTestBase,
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
