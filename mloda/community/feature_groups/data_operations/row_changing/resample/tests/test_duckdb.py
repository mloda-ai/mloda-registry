"""Tests for DuckdbResample compute implementation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

duckdb = pytest.importorskip("duckdb")

from mloda_plugins.compute_framework.base_implementations.duckdb.duckdb_relation import DuckdbRelation

from mloda.community.feature_groups.data_operations.row_changing.resample.duckdb_resample import (
    DuckdbResample,
)
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.duckdb import DuckdbTestMixin
from mloda.testing.feature_groups.data_operations.row_changing.resample.resample import (
    ResampleTestBase,
    _create_resample_arrow_table,
)


class TestDuckdbResample(DuckdbTestMixin, ResampleTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return DuckdbResample


class TestDuckdbResampleNonUtcSession:
    """Resample must produce UTC-anchored bucket labels regardless of session TZ (issue #265).

    This is issue #238 -- already fixed for ``time_bucketization`` -- re-appearing
    in ``resample`` (issue #265). ``DuckdbResample._compute_resample`` floors
    timestamps with ``_floor_expr`` (which emits ``DATE_TRUNC`` for n=1) imported
    from ``time_bucketization``, but it never pins the DuckDB session timezone to
    UTC. ``DATE_TRUNC`` operates in the connection's session timezone, so on a
    non-UTC session a UTC instant gets floored to *local* midnight instead of UTC
    midnight, yielding a misaligned, non-UTC bucket-start label.

    With a 1-day bucket over the whole 12-row fixture (all timestamps on
    2023-01-01 between 08:00 and 10:30 UTC), every row belongs to the single UTC
    day bucket ``2023-01-01 00:00:00 UTC``. On an ``America/New_York`` (UTC-5)
    session WITHOUT the fix, ``DATE_TRUNC`` floors e.g. 2023-01-01 08:05 UTC
    (= 03:05 local) to 2023-01-01 00:00 local = 2023-01-01 05:00 UTC -- a WRONG,
    non-UTC-midnight label -- so this test fails today.

    The test builds its own connection (rather than the mixin's) so it can
    deterministically reproduce the bug by presetting a non-UTC session timezone,
    independent of the host OS timezone.
    """

    def test_resample_1_day_utc_anchored_on_non_utc_session(self) -> None:
        con = duckdb.connect()
        # Preset a NON-UTC session timezone; this is what triggers the bug.
        con.execute("SET TimeZone='America/New_York'")

        rel = DuckdbRelation.from_arrow(con, _create_resample_arrow_table())
        feature_name = "value__resample_1_day_mean"
        fs = make_feature_set(feature_name, partition_by=[], time_column="ts")
        result = DuckdbResample.calculate_feature(rel, fs)

        bucket_col = list(result.to_arrow_table().column("ts").to_pylist())
        value_col = list(result.to_arrow_table().column(feature_name).to_pylist())

        # All 12 rows fall into the single UTC day bucket 2023-01-01 00:00 UTC.
        assert len(bucket_col) == 1, f"expected exactly 1 bucket, got {len(bucket_col)}: {bucket_col!r}"

        actual = bucket_col[0]
        if isinstance(actual, str):
            actual = datetime.fromisoformat(actual)
        expected = datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)
        assert actual == expected, f"bucket label {actual!r} != UTC-midnight {expected!r}"

        # Mean of all NON-NULL values [10, 2, 6, 20, 30, 100, 50, 4, 8] -> 230 / 9.
        assert value_col[0] == pytest.approx(230.0 / 9.0)

    def test_resample_15_minute_utc_anchored_on_non_utc_session(self) -> None:
        # n>1 sub-hour bucket exercises the time_bucket floor branch. On a non-UTC
        # session the floored bucket *label* is tagged with the session zone rather
        # than UTC, so the bucket-start labels diverge from the PyArrow oracle even
        # though the underlying instants line up. Keying the comparison by the ISO
        # string (which carries the offset) makes that label drift observable, so
        # the result must still match the UTC-anchored oracle (issue #265 / #238).
        from mloda.community.feature_groups.data_operations.row_changing.resample.pyarrow_resample import (
            PyArrowResample,
        )
        from mloda.testing.feature_groups.data_operations.helpers import extract_column

        con = duckdb.connect()
        con.execute("SET TimeZone='Asia/Kolkata'")  # UTC+5:30, non-UTC session

        arrow_table = _create_resample_arrow_table()
        rel = DuckdbRelation.from_arrow(con, arrow_table)
        feature_name = "value__resample_15_minute_mean"
        fs = make_feature_set(feature_name, partition_by=[], time_column="ts")

        result = DuckdbResample.calculate_feature(rel, fs)
        oracle = PyArrowResample.calculate_feature(arrow_table, fs)

        def _to_map(buckets: list[Any], vals: list[Any]) -> dict[str, Any]:
            # Key by the ISO string so the bucket-label timezone offset (UTC vs the
            # session zone) participates in the comparison, not just the instant.
            out: dict[str, Any] = {}
            for b, v in zip(buckets, vals):
                key = b if isinstance(b, str) else b.isoformat()
                out[key] = v
            return out

        actual = _to_map(
            list(result.to_arrow_table().column("ts").to_pylist()),
            list(result.to_arrow_table().column(feature_name).to_pylist()),
        )
        expected = _to_map(extract_column(oracle, "ts"), extract_column(oracle, feature_name))

        assert set(actual.keys()) == set(expected.keys()), (
            f"bucket keys diverge: {sorted(actual)} vs {sorted(expected)}"
        )
        for k, exp in expected.items():
            got = actual[k]
            if exp is None:
                assert got is None, f"bucket {k!r}: expected None got {got!r}"
            else:
                assert got == pytest.approx(exp, rel=1e-6), f"bucket {k!r}: {got!r} != {exp!r}"
