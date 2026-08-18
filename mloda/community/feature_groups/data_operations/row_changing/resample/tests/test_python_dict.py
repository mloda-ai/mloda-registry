"""Tests for PythonDictResample compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_changing.resample.python_dict_resample import (
    PythonDictResample,
)
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_changing.resample.resample import (
    ResampleTestBase,
)


class TestPythonDictResample(PythonDictTestMixin, ResampleTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictResample


class TestPythonDictNonDivisorBucketRowCount:
    """``n=7`` does not evenly divide 60 minutes; bucket ANCHORING (not just labeling) diverges.

    ``python_dict_resample._floor_dt`` is a verbatim copy of
    ``python_dict_time_bucketization._floor_dt``, which anchors minute buckets
    at ``(minute // n) * n`` within the enclosing hour. PyArrow's
    ``floor_temporal`` (the reference oracle used by ``_compute_resample``'s
    counterpart ``PyArrowResample``) is epoch-anchored: multiples of the
    bucket duration since 1970-01-01 UTC. The two schemes only coincide when
    ``n`` evenly divides 60.

    For rows at 10:03 and 10:09, PyArrow's epoch anchor floors BOTH into the
    same [10:01, 10:08) bucket (they get summed together), while the naive
    enclosing-hour anchor floors them into two DIFFERENT buckets ([10:00,
    10:07) and [10:07, 10:14)), so they stay separate. This changes the
    output ROW COUNT, not just a bucket label, so ``PythonDictResample``
    must match the live PyArrow oracle exactly (both row count and the
    per-bucket sums), not a hand-computed expectation.
    """

    def test_7_minute_sum_row_count_and_buckets_match_pyarrow_oracle(self) -> None:
        from datetime import datetime

        import pyarrow as pa

        from mloda.community.feature_groups.data_operations.row_changing.resample.pyarrow_resample import (
            PyArrowResample,
        )
        from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set

        arrow_table = pa.table(
            {
                "ts": pa.array(
                    [
                        datetime(2023, 1, 1, 10, 3, 0),
                        datetime(2023, 1, 1, 10, 9, 0),
                        datetime(2023, 1, 1, 10, 14, 0),
                    ],
                    type=pa.timestamp("us", tz="UTC"),
                ),
                "v": pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            }
        )
        data = {name: arrow_table.column(name).to_pylist() for name in arrow_table.column_names}
        fs = make_feature_set("v__resample_7_minute_sum", time_column="ts")

        result = PythonDictResample.calculate_feature(data, fs)
        oracle = PyArrowResample.calculate_feature(arrow_table, fs)

        result_ts = extract_column(result, "ts")
        oracle_ts = extract_column(oracle, "ts")
        assert len(result_ts) == len(oracle_ts), (
            f"row count {len(result_ts)} != PyArrow oracle row count {len(oracle_ts)}: "
            f"python_dict buckets={result_ts!r}, oracle buckets={oracle_ts!r}"
        )

        result_map = dict(zip(result_ts, extract_column(result, "v__resample_7_minute_sum")))
        oracle_map = dict(zip(oracle_ts, extract_column(oracle, "v__resample_7_minute_sum")))
        assert result_map == oracle_map, f"{result_map!r} != oracle {oracle_map!r}"


class TestPythonDictNullTimestamp:
    """A null ``time_column`` value must not crash resample.

    ``PythonDictResample._compute_resample`` calls ``_floor_dt(value, n, unit)``
    for every row's timestamp with no null check. ``_floor_dt(None, ...)``
    calls ``_wall_clock_epoch_seconds(None)``, which does
    ``dt.replace(tzinfo=timezone.utc)`` and raises
    ``AttributeError: 'NoneType' object has no attribute 'replace'``.

    PyArrow (the live reference oracle) does not crash:
    ``pyarrow.compute.floor_temporal`` propagates the null through to its own
    bucket, i.e. a non-empty bucket whose ``time_column`` key is ``None``,
    aggregating whatever rows share that null bucket.
    """

    def test_null_timestamp_matches_pyarrow_oracle(self) -> None:
        from datetime import datetime, timezone

        import pyarrow as pa

        from mloda.community.feature_groups.data_operations.row_changing.resample.pyarrow_resample import (
            PyArrowResample,
        )
        from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set

        u = timezone.utc
        arrow_table = pa.table(
            {
                "region": pa.array(["A", "A", "A", "B"], type=pa.string()),
                "ts": pa.array(
                    [
                        datetime(2023, 1, 1, 10, 5, 0, tzinfo=u),
                        None,
                        datetime(2023, 1, 1, 10, 40, 0, tzinfo=u),
                        datetime(2023, 1, 1, 11, 0, 0, tzinfo=u),
                    ],
                    type=pa.timestamp("us", tz="UTC"),
                ),
                "value": pa.array([10.0, 20.0, 30.0, 40.0], type=pa.float64()),
            }
        )
        data = {name: arrow_table.column(name).to_pylist() for name in arrow_table.column_names}
        fs = make_feature_set("value__resample_1_hour_mean", partition_by=["region"], time_column="ts")

        result = PythonDictResample.calculate_feature(data, fs)
        oracle = PyArrowResample.calculate_feature(arrow_table, fs)

        def _bucket_map(res: Any) -> dict[tuple[Any, Any], Any]:
            regions = extract_column(res, "region")
            buckets = extract_column(res, "ts")
            values = extract_column(res, "value__resample_1_hour_mean")
            return {(regions[i], buckets[i]): values[i] for i in range(len(regions))}

        result_map = _bucket_map(result)
        oracle_map = _bucket_map(oracle)
        assert result_map == oracle_map, f"{result_map!r} != PyArrow oracle {oracle_map!r}"
