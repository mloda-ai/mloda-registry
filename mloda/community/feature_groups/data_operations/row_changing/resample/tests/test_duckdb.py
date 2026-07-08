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
    """Resample stays UTC-anchored on a non-UTC session, via the core chokepoint.

    Presets a non-UTC session, then relies on the core UTC guarantee (not a
    per-FG pin) to anchor the bucket labels.
    """

    def test_resample_1_day_utc_anchored_on_non_utc_session(self) -> None:
        from mloda.testing.feature_groups.data_operations.mixins.duckdb import pin_connection_utc_via_core

        con = duckdb.connect()
        con.execute("SET TimeZone='America/New_York'")
        pin_connection_utc_via_core(con)

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
        # n>1 sub-hour bucket exercises the time_bucket floor branch.
        from mloda.community.feature_groups.data_operations.row_changing.resample.pyarrow_resample import (
            PyArrowResample,
        )
        from mloda.testing.feature_groups.data_operations.helpers import extract_column
        from mloda.testing.feature_groups.data_operations.mixins.duckdb import pin_connection_utc_via_core

        con = duckdb.connect()
        con.execute("SET TimeZone='Asia/Kolkata'")  # UTC+5:30, non-UTC session
        pin_connection_utc_via_core(con)

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
