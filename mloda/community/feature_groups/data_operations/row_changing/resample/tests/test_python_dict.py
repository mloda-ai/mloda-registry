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
    """``n=7`` does not evenly divide 60 minutes, so bucket ANCHORING (not just labeling) matters.

    ``python_dict_resample._floor_dt`` and PyArrow's ``floor_temporal`` oracle both anchor
    minute buckets to multiples of the bucket duration since 1970-01-01 UTC, not to the
    enclosing hour; a naive enclosing-hour anchor would instead split rows 10:03 and 10:09
    into two different buckets rather than summing them together, changing the output ROW
    COUNT. This regression guard pins the two floors staying in sync against the live
    PyArrow oracle, not a hand-computed expectation.
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

    ``PythonDictResample._compute_resample`` guards each row with
    ``None if value is None else _floor_dt(value, n, unit)`` before flooring, so a null
    timestamp bypasses ``_floor_dt`` entirely instead of reaching the ``.replace(tzinfo=...)``
    call that would otherwise raise on ``None``. PyArrow's ``floor_temporal`` oracle also
    doesn't crash: it propagates the null through to its own bucket, aggregating whatever
    rows share that null bucket. This test pins that ``PythonDictResample`` matches it.
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


class TestPythonDictMinMaxSkipsNan:
    """min/max must skip a NaN value within a bucket, not propagate it.

    ``reduce_agg`` skips NaN for ``min``/``max``, matching PyArrow's ``group_by().aggregate()``
    (the reference oracle ``PyArrowResample`` also uses) and every other min/max-skips-NaN
    aggregation in this codebase.
    """

    @staticmethod
    def _hour_bucket_via(agg: str) -> tuple[list[Any], list[Any]]:
        from datetime import datetime, timezone

        import pyarrow as pa

        from mloda.community.feature_groups.data_operations.row_changing.resample.pyarrow_resample import (
            PyArrowResample,
        )
        from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set

        u = timezone.utc
        arrow_table = pa.table(
            {
                "ts": pa.array(
                    [
                        datetime(2023, 1, 1, 10, 0, 0, tzinfo=u),
                        datetime(2023, 1, 1, 10, 5, 0, tzinfo=u),
                        datetime(2023, 1, 1, 10, 10, 0, tzinfo=u),
                    ],
                    type=pa.timestamp("us", tz="UTC"),
                ),
                "v": pa.array([float("nan"), 1.0, 3.0], type=pa.float64()),
            }
        )
        data = {name: arrow_table.column(name).to_pylist() for name in arrow_table.column_names}
        fs = make_feature_set(f"v__resample_1_hour_{agg}", time_column="ts")

        result = PythonDictResample.calculate_feature(data, fs)
        oracle = PyArrowResample.calculate_feature(arrow_table, fs)

        return (
            extract_column(result, f"v__resample_1_hour_{agg}"),
            extract_column(oracle, f"v__resample_1_hour_{agg}"),
        )

    def test_min_skips_nan_matches_pyarrow_oracle(self) -> None:
        result_col, oracle_col = self._hour_bucket_via("min")
        assert oracle_col == [1.0], f"sanity check on the oracle itself failed: {oracle_col!r}"
        assert result_col == oracle_col, f"{result_col!r} != PyArrow oracle {oracle_col!r}"

    def test_max_skips_nan_matches_pyarrow_oracle(self) -> None:
        result_col, oracle_col = self._hour_bucket_via("max")
        assert oracle_col == [3.0], f"sanity check on the oracle itself failed: {oracle_col!r}"
        assert result_col == oracle_col, f"{result_col!r} != PyArrow oracle {oracle_col!r}"


class TestPythonDictNanPartitionKeyGrouping:
    """A NaN partition-key value must group with itself, not split into singleton buckets.

    ``group_key_value`` maps a NaN partition value to a shared sentinel before it becomes
    part of the group key, so two rows that both carry ``float('nan')`` (which compares
    unequal to itself) still land in the same bucket. ``PyArrowResample`` groups via
    PyArrow's own ``Table.group_by()``, which merges all NaN keys of a column into a
    single group, so it is a valid live oracle here.
    """

    def test_nan_partition_rows_merge_into_one_bucket_matches_pyarrow_oracle(self) -> None:
        from datetime import datetime, timezone

        import pyarrow as pa

        from mloda.community.feature_groups.data_operations.row_changing.resample.pyarrow_resample import (
            PyArrowResample,
        )
        from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set

        u = timezone.utc
        arrow_table = pa.table(
            {
                "grp": pa.array([float("nan"), float("nan")], type=pa.float64()),
                "ts": pa.array(
                    [
                        datetime(2023, 1, 1, 10, 0, 0, tzinfo=u),
                        datetime(2023, 1, 1, 10, 5, 0, tzinfo=u),
                    ],
                    type=pa.timestamp("us", tz="UTC"),
                ),
                "v": pa.array([1.0, 2.0], type=pa.float64()),
            }
        )
        data = {name: arrow_table.column(name).to_pylist() for name in arrow_table.column_names}
        fs = make_feature_set("v__resample_1_hour_sum", partition_by=["grp"], time_column="ts")

        result = PythonDictResample.calculate_feature(data, fs)
        oracle = PyArrowResample.calculate_feature(arrow_table, fs)

        oracle_ts = extract_column(oracle, "ts")
        assert len(oracle_ts) == 1, f"sanity check on the oracle itself failed: expected 1 row, got {oracle_ts!r}"

        result_ts = extract_column(result, "ts")
        assert len(result_ts) == 1, (
            f"expected the two NaN-partition rows to merge into one bucket (matching the "
            f"PyArrow oracle's single row), got {len(result_ts)} rows: {result_ts!r}"
        )
        assert extract_column(result, "v__resample_1_hour_sum") == [3.0]


class TestPythonDictNanPartitionColumnLeaksGroupKeySentinel:
    """DEFECT A (regression): the output partition column must hold the original NaN value,
    never the internal ``group_key_value`` sentinel used only to build the group key.

    ``_compute_resample`` normalizes group keys via ``group_key_value`` (NaN -> a shared
    sentinel object) so NaN-valued rows land in one bucket, but then emits the output
    partition column straight from ``key[:-1]``, the NORMALIZED key, leaking the opaque
    sentinel into user-visible data. ``python_dict_aggregation.py`` shows the correct
    pattern: emit ``col[first_idx]``, the raw representative value from the original
    column, not the key component.
    """

    def test_partition_column_holds_raw_nan_not_the_group_key_sentinel(self) -> None:
        import math
        from datetime import datetime, timezone

        t0 = datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        data: dict[str, list[Any]] = {
            "grp": [float("nan"), float("nan"), 1.0],
            "ts": [t0, t0, t0],
            "v": [1.0, 2.0, 5.0],
        }

        result = PythonDictResample._compute_resample(data, "s", "v", "ts", ["grp"], 1, "hour", "sum")

        assert all(isinstance(v, float) for v in result["grp"]), (
            f"expected the partition column to hold plain float values, got a leaked "
            f"non-float sentinel: {result['grp']!r}"
        )
        assert math.isnan(result["grp"][0]), (
            f"expected the NaN group's partition value to be the original NaN, got "
            f"{result['grp'][0]!r} (the internal group_key_value sentinel leaked into the "
            f"output instead of the raw column value)"
        )
        assert result["grp"][1] == 1.0
        assert result["s"] == [3.0, 5.0]

    def test_partition_column_matches_pyarrow_oracle(self) -> None:
        import math
        from datetime import datetime, timezone

        import pyarrow as pa

        from mloda.community.feature_groups.data_operations.row_changing.resample.pyarrow_resample import (
            PyArrowResample,
        )
        from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set

        u = timezone.utc
        arrow_table = pa.table(
            {
                "grp": pa.array([float("nan"), float("nan"), 1.0], type=pa.float64()),
                "ts": pa.array([datetime(2023, 1, 1, 10, 0, 0, tzinfo=u)] * 3, type=pa.timestamp("us", tz="UTC")),
                "v": pa.array([1.0, 2.0, 5.0], type=pa.float64()),
            }
        )
        data = {name: arrow_table.column(name).to_pylist() for name in arrow_table.column_names}
        fs = make_feature_set("v__resample_1_hour_sum", partition_by=["grp"], time_column="ts")

        result = PythonDictResample.calculate_feature(data, fs)
        oracle = PyArrowResample.calculate_feature(arrow_table, fs)

        oracle_grp = extract_column(oracle, "grp")
        assert any(isinstance(v, float) and math.isnan(v) for v in oracle_grp), (
            f"sanity check on the oracle itself failed: {oracle_grp!r}"
        )

        result_grp = extract_column(result, "grp")
        assert all(v is None or isinstance(v, float) for v in result_grp), (
            f"expected the partition column to hold plain float/None values, got a leaked "
            f"non-float sentinel: {result_grp!r}"
        )
        assert any(isinstance(v, float) and math.isnan(v) for v in result_grp), (
            f"expected one partition value to be the original NaN (matching the PyArrow oracle), got {result_grp!r}"
        )
