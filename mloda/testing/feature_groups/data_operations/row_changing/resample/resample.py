"""Shared test base, dedicated fixture, and expected values for resample tests.

``resample`` collapses event rows onto a regular time grid. Each row's
``time_column`` is floored to the start of its ``n*unit`` bucket (epoch-anchored
fixed-freq floor, IDENTICAL to ``time_bucketization``'s floor for
minute / hour / day so every backend agrees). Rows are then grouped by
``(*partition_by, bucket_start)`` and the source value column is aggregated with
one of the ORDER-INDEPENDENT aggregations ``mean / sum / count / min / max``.

This operation CHANGES the row count: the output has exactly one row per
NON-EMPTY ``(partition, bucket)`` pair (a bucket that has >= 1 input row). Empty
gap buckets are NOT emitted. A bucket that has rows but whose source values are
ALL null still emits a row -- it is non-empty -- with ``count = 0`` and
``mean / sum / min / max = None``.

Because the output row count differs from the input and the output row order is
not guaranteed, the standard ``DataOpsTestBase._compare_with_reference`` (which
assumes equal row counts and row-by-row equality) does NOT apply. Instead this
module mirrors the AGGREGATION test base: it builds a result map keyed by
``(partition..., bucket_start)`` and compares it to a pinned ``EXPECTED_*`` dict,
and separately asserts the output row count equals the number of non-empty
buckets (the row-CHANGING assertion).

Expected-value literals were computed offline from PyArrow as the oracle
(``pyarrow.compute.floor_temporal`` then ``pa.TableGroupBy.aggregate``) and
pasted in. PyArrow is the live reference (``PyArrowResample``); pinning to the
PyArrow oracle (NOT raw pandas) matters because the two diverge on the all-null
``sum`` bucket: pandas returns ``0.0`` there, PyArrow returns ``None`` -- and the
cross-framework test compares every backend, including PyArrow, against
``PyArrowResample``, so the pinned literal must equal the oracle's.

Fixture row layout (``id`` is a witness column; 12 rows, 2 interleaved
partitions ``region`` A / B)::

    id | region | ts (UTC)            | value | Purpose
    ---+--------+--------------------+-------+-----------------------------------
    0  | A      | 08:05:00           | 10.0  | A hour08 bucket
    1  | B      | 08:10:00           | 2.0   | B hour08
    2  | A      | 08:50:00           | None  | A hour08 (null excluded from mean/sum/count)
    3  | B      | 08:40:00           | 6.0   | B hour08
    4  | A      | 09:15:00           | 20.0  | A hour09
    5  | B      | 09:05:00           | None  | B hour09 (null)
    6  | A      | 09:45:00           | 30.0  | A hour09
    7  | B      | 09:50:00           | None  | B hour09 (null) -> B hour09 ALL-NULL bucket
    8  | A      | 10:00:00           | 100.0 | A hour10
    9  | B      | 10:30:00           | 50.0  | B hour10
    10 | A      | 08:20:00           | 4.0   | A hour08 3rd row (15-min: 08:15 bucket)
    11 | B      | 08:12:00           | 8.0   | B hour08 (15-min: 08:00 bucket)

    All timestamps fall on 2023-01-01. Multiple rows share an hour bucket
    within each partition (tests aggregation); buckets span several hours
    (tests row-count reduction: 12 input rows -> 6 non-empty 1-hour buckets per
    region). The B hour09 bucket has two rows, both null, so it emits with
    count=0 / mean=None (all-null non-empty bucket spec).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.helpers import extract_column as _extract_column
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda.user import Feature

_U = timezone.utc


# ---------------------------------------------------------------------------
# Dedicated 12-row fixture (UTC timestamps, two interleaved partitions)
# ---------------------------------------------------------------------------

_RESAMPLE_IDS: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

_RESAMPLE_REGIONS: list[str] = ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]

_RESAMPLE_TIMESTAMPS: list[datetime] = [
    datetime(2023, 1, 1, 8, 5, 0, tzinfo=_U),  # 0  A hour08
    datetime(2023, 1, 1, 8, 10, 0, tzinfo=_U),  # 1  B hour08
    datetime(2023, 1, 1, 8, 50, 0, tzinfo=_U),  # 2  A hour08
    datetime(2023, 1, 1, 8, 40, 0, tzinfo=_U),  # 3  B hour08
    datetime(2023, 1, 1, 9, 15, 0, tzinfo=_U),  # 4  A hour09
    datetime(2023, 1, 1, 9, 5, 0, tzinfo=_U),  # 5  B hour09
    datetime(2023, 1, 1, 9, 45, 0, tzinfo=_U),  # 6  A hour09
    datetime(2023, 1, 1, 9, 50, 0, tzinfo=_U),  # 7  B hour09 (all-null bucket w/ row 5)
    datetime(2023, 1, 1, 10, 0, 0, tzinfo=_U),  # 8  A hour10
    datetime(2023, 1, 1, 10, 30, 0, tzinfo=_U),  # 9  B hour10
    datetime(2023, 1, 1, 8, 20, 0, tzinfo=_U),  # 10 A hour08 (15-min: 08:15)
    datetime(2023, 1, 1, 8, 12, 0, tzinfo=_U),  # 11 B hour08 (15-min: 08:00)
]

_RESAMPLE_VALUES: list[float | None] = [
    10.0,  # 0  A
    2.0,  # 1  B
    None,  # 2  A null
    6.0,  # 3  B
    20.0,  # 4  A
    None,  # 5  B null
    30.0,  # 6  A
    None,  # 7  B null -> B hour09 all-null
    100.0,  # 8  A
    50.0,  # 9  B
    4.0,  # 10 A
    8.0,  # 11 B
]


def _create_resample_arrow_table() -> pa.Table:
    """Create the 12-row PyArrow table used by every resample test."""
    return pa.table(
        {
            "id": pa.array(_RESAMPLE_IDS, type=pa.int64()),
            "region": pa.array(_RESAMPLE_REGIONS, type=pa.string()),
            "ts": pa.array(_RESAMPLE_TIMESTAMPS, type=pa.timestamp("us", tz="UTC")),
            "value": pa.array(_RESAMPLE_VALUES, type=pa.float64()),
        }
    )


# Convenience bucket-start constructors.
def _h(hour: int, minute: int = 0) -> datetime:
    return datetime(2023, 1, 1, hour, minute, 0, tzinfo=_U)


# ---------------------------------------------------------------------------
# Expected values (offline-computed from PyArrow oracle; keyed (region, bucket))
# ---------------------------------------------------------------------------
# pyarrow.compute.floor_temporal(ts, multiple=n, unit=unit) then
# pa.Table.group_by([*partition, bucket]).aggregate([("value", <agg>)]).
# The B hour09 bucket is non-empty (rows 5 & 7) but all-null: sum/mean=None,
# count=0. pandas would report sum=0.0 there; we pin the PyArrow value None.

EXPECTED_1_HOUR_MEAN: dict[tuple[Any, ...], Any] = {
    ("A", _h(8)): 7.0,  # [10.0, None, 4.0] -> 14/2
    ("A", _h(9)): 25.0,  # [20.0, 30.0]
    ("A", _h(10)): 100.0,  # [100.0]
    ("B", _h(8)): 16.0 / 3.0,  # [2.0, 6.0, 8.0] -> 16/3
    ("B", _h(9)): None,  # [None, None] all-null
    ("B", _h(10)): 50.0,  # [50.0]
}

EXPECTED_1_HOUR_SUM: dict[tuple[Any, ...], Any] = {
    ("A", _h(8)): 14.0,
    ("A", _h(9)): 50.0,
    ("A", _h(10)): 100.0,
    ("B", _h(8)): 16.0,
    ("B", _h(9)): None,  # PyArrow: all-null sum is None (pandas would say 0.0)
    ("B", _h(10)): 50.0,
}

EXPECTED_1_HOUR_COUNT: dict[tuple[Any, ...], Any] = {
    ("A", _h(8)): 2,  # 10.0, 4.0 (null excluded)
    ("A", _h(9)): 2,
    ("A", _h(10)): 1,
    ("B", _h(8)): 3,
    ("B", _h(9)): 0,  # both null -> count 0, but bucket still emits
    ("B", _h(10)): 1,
}

EXPECTED_15_MINUTE_MEAN: dict[tuple[Any, ...], Any] = {
    ("A", _h(8, 0)): 10.0,  # 08:05 -> 08:00 bucket
    ("A", _h(8, 15)): 4.0,  # 08:20 -> 08:15 bucket
    ("A", _h(8, 45)): None,  # 08:50 -> 08:45 bucket, value None
    ("A", _h(9, 15)): 20.0,  # 09:15
    ("A", _h(9, 45)): 30.0,  # 09:45
    ("A", _h(10, 0)): 100.0,
    ("B", _h(8, 0)): 5.0,  # 08:10, 08:12 -> [2.0, 8.0] -> 5.0
    ("B", _h(8, 30)): 6.0,  # 08:40 -> 08:30 bucket
    ("B", _h(9, 0)): None,  # 09:05 -> 09:00 bucket, value None
    ("B", _h(9, 45)): None,  # 09:50 -> 09:45 bucket, value None
    ("B", _h(10, 30)): 50.0,
}

# Whole-table 1-hour mean (no partition). Key is a 1-tuple ``(bucket,)``.
EXPECTED_1_HOUR_MEAN_WHOLE: dict[tuple[Any, ...], Any] = {
    (_h(8),): 6.0,  # [10, None, 2, 6, 4, 8] -> 30/5
    (_h(9),): 25.0,  # [20, None, 30, None] -> 50/2
    (_h(10),): 75.0,  # [100, 50] -> 75
}


# ---------------------------------------------------------------------------
# Reusable test base class
# ---------------------------------------------------------------------------


class ResampleTestBase(DataOpsTestBase):
    """Abstract base class for resample framework tests.

    Subclasses combine this with a framework mixin (``PyArrowTestMixin``,
    ``PandasTestMixin``, etc.) and a one-liner ``implementation_class``
    classmethod returning the framework-specific feature group.

    The expected base feature-group class is ``ResampleFeatureGroup`` and the
    backends are ``PyArrowResample`` (the reference oracle), ``PandasResample``,
    ``PolarsLazyResample`` and ``DuckdbResample``. SQLite is deferred to a
    later version and is intentionally not tested here.
    """

    @classmethod
    def reference_implementation_class(cls) -> Any:
        # Lazy import so this testing module imports cleanly even before the
        # production code exists (Red phase). The per-backend test files import
        # their implementation classes at module top level; that is where Red
        # fails with ModuleNotFoundError.
        from mloda.community.feature_groups.data_operations.row_changing.resample.pyarrow_resample import (
            PyArrowResample,
        )

        return PyArrowResample

    # -- Setup: use the dedicated 12-row resample fixture -------------------

    def setup_method(self) -> None:
        """Override the canonical-fixture setup to use the dedicated 12-row table."""
        super().setup_method()  # connections + canonical data (mostly unused)
        self._arrow_table = _create_resample_arrow_table()
        self.test_data = self.create_test_data(self._arrow_table)

    # -- Result-map helpers -------------------------------------------------

    @staticmethod
    def _normalize_bucket_key(value: Any) -> Any:
        """Coerce a bucket-start value to a tz-aware ``datetime`` for dict keys.

        Different backends return different timestamp types: PyArrow / polars /
        duckdb return ``datetime``; pandas returns ``pd.Timestamp`` (a datetime
        subclass, but normalised here for safe dict-key identity); a TEXT-based
        backend would return an ISO string. Mirrors how the time_bucketization
        harness accepts ISO strings from SQLite.
        """
        if value is None:
            return None
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        # pd.Timestamp -> datetime (pd.Timestamp.to_pydatetime exists; plain
        # datetime passes through unchanged).
        to_py = getattr(value, "to_pydatetime", None)
        if to_py is not None:
            return to_py()
        return value

    def _build_resample_map(
        self,
        result: Any,
        feature_name: str,
        time_column: str,
        partition_by: list[str],
    ) -> dict[tuple[Any, ...], Any]:
        """Build a ``{(partition..., bucket_start): agg_value}`` map from a result."""
        bucket_col = [self._normalize_bucket_key(v) for v in self.extract_column(result, time_column)]
        agg_col = self.extract_column(result, feature_name)
        partition_cols = [self.extract_column(result, p) for p in partition_by]

        out: dict[tuple[Any, ...], Any] = {}
        for i in range(len(bucket_col)):
            key = tuple(partition_cols[j][i] for j in range(len(partition_by))) + (bucket_col[i],)
            out[key] = agg_col[i]
        return out

    def _assert_map_equals(
        self,
        actual: dict[tuple[Any, ...], Any],
        expected: dict[tuple[Any, ...], Any],
        *,
        use_approx: bool = False,
    ) -> None:
        assert len(actual) == len(expected), f"bucket count {len(actual)} != expected {len(expected)}"
        for key, exp in expected.items():
            assert key in actual, f"missing bucket key {key!r}; got keys {sorted(map(str, actual))}"
            got = actual[key]
            if exp is None:
                assert got is None, f"bucket {key!r}: expected None, got {got!r}"
            elif use_approx:
                assert got == pytest.approx(exp, rel=1e-6), f"bucket {key!r}: {got!r} != {exp!r}"
            else:
                assert got == exp, f"bucket {key!r}: {got!r} != {exp!r}"

    def _resample_fs(self, feature_name: str, partition_by: list[str]) -> FeatureSet:
        return make_feature_set(feature_name, partition_by=partition_by, time_column="ts")

    # -- Core per-partition value tests -------------------------------------

    def test_1_hour_mean_per_partition(self) -> None:
        """1-hour mean per region. Output row count == number of non-empty buckets (6)."""
        feature_name = "value__resample_1_hour_mean"
        fs = self._resample_fs(feature_name, ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        # Row-CHANGING assertion: one row per non-empty (region, hour) bucket.
        assert self.get_row_count(result) == len(EXPECTED_1_HOUR_MEAN)

        result_map = self._build_resample_map(result, feature_name, "ts", ["region"])
        self._assert_map_equals(result_map, EXPECTED_1_HOUR_MEAN, use_approx=True)

    def test_1_hour_sum_per_partition(self) -> None:
        feature_name = "value__resample_1_hour_sum"
        fs = self._resample_fs(feature_name, ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_map = self._build_resample_map(result, feature_name, "ts", ["region"])
        self._assert_map_equals(result_map, EXPECTED_1_HOUR_SUM, use_approx=True)

    def test_1_hour_count_per_partition(self) -> None:
        feature_name = "value__resample_1_hour_count"
        fs = self._resample_fs(feature_name, ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_map = self._build_resample_map(result, feature_name, "ts", ["region"])
        self._assert_map_equals(result_map, EXPECTED_1_HOUR_COUNT)

    def test_15_minute_mean_per_partition(self) -> None:
        """n>1 / sub-hour unit: guards epoch-anchored bucket alignment (:00/:15/:30/:45)."""
        feature_name = "value__resample_15_minute_mean"
        fs = self._resample_fs(feature_name, ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert self.get_row_count(result) == len(EXPECTED_15_MINUTE_MEAN)
        result_map = self._build_resample_map(result, feature_name, "ts", ["region"])
        self._assert_map_equals(result_map, EXPECTED_15_MINUTE_MEAN, use_approx=True)

    def test_whole_table_1_hour_mean(self) -> None:
        """No partition (partition_by=[]): one row per hour bucket across the whole table."""
        feature_name = "value__resample_1_hour_mean"
        fs = self._resample_fs(feature_name, [])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert self.get_row_count(result) == len(EXPECTED_1_HOUR_MEAN_WHOLE)
        result_map = self._build_resample_map(result, feature_name, "ts", [])
        self._assert_map_equals(result_map, EXPECTED_1_HOUR_MEAN_WHOLE, use_approx=True)

    # -- Row-changing semantics ---------------------------------------------

    def test_row_count_changes(self) -> None:
        """Output rows == number of non-empty buckets AND differs from input rows (12)."""
        feature_name = "value__resample_1_hour_mean"
        fs = self._resample_fs(feature_name, ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        n_buckets = len(EXPECTED_1_HOUR_MEAN)
        assert self.get_row_count(result) == n_buckets
        assert self.get_row_count(result) != len(_RESAMPLE_IDS), (
            "resample must change the row count (input had 12 rows)"
        )

    def test_no_empty_gap_buckets_emitted(self) -> None:
        """Only non-empty buckets are emitted (no gap-fill).

        A hour has no row at, e.g., ('A', 11:00); that bucket must be ABSENT.
        """
        feature_name = "value__resample_1_hour_mean"
        fs = self._resample_fs(feature_name, ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_map = self._build_resample_map(result, feature_name, "ts", ["region"])
        assert ("A", _h(11)) not in result_map
        assert ("B", _h(11)) not in result_map
        assert set(result_map.keys()) == set(EXPECTED_1_HOUR_MEAN.keys())

    def test_bucket_start_labels_floored(self) -> None:
        """Bucket-start labels are floored to the bucket start, not the raw timestamp.

        Spot-check: ('A', hour08) collapses rows at 08:05 / 08:50 / 08:20 to the
        single key 08:00:00, and ('B', 08:30) for the 15-minute case (08:40 row).
        """
        feature_name = "value__resample_1_hour_mean"
        fs = self._resample_fs(feature_name, ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_map = self._build_resample_map(result, feature_name, "ts", ["region"])
        assert ("A", _h(8)) in result_map
        # The raw (unfloored) timestamps must NOT appear as bucket keys.
        assert ("A", _h(8, 5)) not in result_map
        assert ("A", _h(8, 50)) not in result_map

    # -- Null handling ------------------------------------------------------

    def test_null_excluded_from_mean_and_sum(self) -> None:
        """A null source value is excluded from mean/sum within its bucket.

        ('A', hour08) holds [10.0, None, 4.0]: mean = 14/2 = 7.0, sum = 14.0
        (the null at row 2 is skipped).
        """
        mean_fs = self._resample_fs("value__resample_1_hour_mean", ["region"])
        mean_result = self.implementation_class().calculate_feature(self.test_data, mean_fs)
        mean_map = self._build_resample_map(mean_result, "value__resample_1_hour_mean", "ts", ["region"])
        assert mean_map[("A", _h(8))] == pytest.approx(7.0)

        sum_fs = self._resample_fs("value__resample_1_hour_sum", ["region"])
        sum_result = self.implementation_class().calculate_feature(self.test_data, sum_fs)
        sum_map = self._build_resample_map(sum_result, "value__resample_1_hour_sum", "ts", ["region"])
        assert sum_map[("A", _h(8))] == pytest.approx(14.0)

    def test_count_reflects_non_null(self) -> None:
        """count counts NON-NULL values: ('A', hour08) = 2 (the null is not counted)."""
        fs = self._resample_fs("value__resample_1_hour_count", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        count_map = self._build_resample_map(result, "value__resample_1_hour_count", "ts", ["region"])
        assert count_map[("A", _h(8))] == 2

    def test_all_null_bucket_emits_with_count_zero(self) -> None:
        """An all-null but non-empty bucket emits a row: count=0, mean/sum=None.

        ('B', hour09) has rows 5 & 7, both null. It is non-empty, so it must be
        present in the output with count 0 and mean / sum None.
        """
        count_fs = self._resample_fs("value__resample_1_hour_count", ["region"])
        count_result = self.implementation_class().calculate_feature(self.test_data, count_fs)
        count_map = self._build_resample_map(count_result, "value__resample_1_hour_count", "ts", ["region"])
        assert ("B", _h(9)) in count_map, "all-null bucket must still emit a row"
        assert count_map[("B", _h(9))] == 0

        mean_fs = self._resample_fs("value__resample_1_hour_mean", ["region"])
        mean_result = self.implementation_class().calculate_feature(self.test_data, mean_fs)
        mean_map = self._build_resample_map(mean_result, "value__resample_1_hour_mean", "ts", ["region"])
        assert mean_map[("B", _h(9))] is None

        sum_fs = self._resample_fs("value__resample_1_hour_sum", ["region"])
        sum_result = self.implementation_class().calculate_feature(self.test_data, sum_fs)
        sum_map = self._build_resample_map(sum_result, "value__resample_1_hour_sum", "ts", ["region"])
        assert sum_map[("B", _h(9))] is None

    def test_min_max_skip_nulls(self) -> None:
        """min / max skip nulls; ('A', hour08) over [10.0, None, 4.0] -> min 4.0, max 10.0."""
        min_fs = self._resample_fs("value__resample_1_hour_min", ["region"])
        min_result = self.implementation_class().calculate_feature(self.test_data, min_fs)
        min_map = self._build_resample_map(min_result, "value__resample_1_hour_min", "ts", ["region"])
        assert min_map[("A", _h(8))] == pytest.approx(4.0)

        max_fs = self._resample_fs("value__resample_1_hour_max", ["region"])
        max_result = self.implementation_class().calculate_feature(self.test_data, max_fs)
        max_map = self._build_resample_map(max_result, "value__resample_1_hour_max", "ts", ["region"])
        assert max_map[("A", _h(8))] == pytest.approx(10.0)

    # -- New column / type ---------------------------------------------------

    def test_new_column_named_exactly(self) -> None:
        """The aggregated column must be named exactly ``{col}__resample_{n}_{unit}_{agg}``."""
        feature_name = "value__resample_1_hour_mean"
        fs = self._resample_fs(feature_name, ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, feature_name)
        assert len(col) == len(EXPECTED_1_HOUR_MEAN)

    def test_result_has_correct_type(self) -> None:
        feature_name = "value__resample_1_hour_mean"
        fs = self._resample_fs(feature_name, ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert isinstance(result, self.get_expected_type())

    # -- Option-based configuration -----------------------------------------

    def test_option_based_config(self) -> None:
        """Option-based configuration (time_column / partition_by / in_features) matches."""
        feature = Feature(
            "value__resample_1_hour_mean",
            options=Options(
                context={
                    "in_features": "value",
                    "time_column": "ts",
                    "partition_by": ["region"],
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_map = self._build_resample_map(result, "value__resample_1_hour_mean", "ts", ["region"])
        self._assert_map_equals(result_map, EXPECTED_1_HOUR_MEAN, use_approx=True)

    # -- Cross-framework comparison (map-based vs PyArrow oracle) ------------

    def _compare_resample_with_reference(
        self,
        feature_name: str,
        partition_by: list[str],
        *,
        use_approx: bool = False,
    ) -> None:
        """Run the feature on this framework and the PyArrow reference; assert maps equal."""
        fs = self._resample_fs(feature_name, partition_by)
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        ref = self.reference_implementation_class().calculate_feature(self._arrow_table, fs)

        result_map = self._build_resample_map(result, feature_name, "ts", partition_by)
        # Reference is a PyArrow table; build its map with the standalone helper.
        ref_bucket = [self._normalize_bucket_key(v) for v in _extract_column(ref, "ts")]
        ref_agg = _extract_column(ref, feature_name)
        ref_partition = [_extract_column(ref, p) for p in partition_by]
        ref_map: dict[tuple[Any, ...], Any] = {}
        for i in range(len(ref_bucket)):
            key = tuple(ref_partition[j][i] for j in range(len(partition_by))) + (ref_bucket[i],)
            ref_map[key] = ref_agg[i]

        self._assert_map_equals(result_map, ref_map, use_approx=use_approx)

    def test_cross_framework_1_hour_mean(self) -> None:
        self._compare_resample_with_reference("value__resample_1_hour_mean", ["region"], use_approx=True)

    def test_cross_framework_1_hour_sum(self) -> None:
        self._compare_resample_with_reference("value__resample_1_hour_sum", ["region"], use_approx=True)

    def test_cross_framework_1_hour_count(self) -> None:
        self._compare_resample_with_reference("value__resample_1_hour_count", ["region"])

    def test_cross_framework_15_minute_mean(self) -> None:
        self._compare_resample_with_reference("value__resample_15_minute_mean", ["region"], use_approx=True)

    def test_cross_framework_whole_table(self) -> None:
        self._compare_resample_with_reference("value__resample_1_hour_mean", [], use_approx=True)

    # -- Error / validation --------------------------------------------------

    def test_bad_unit_rejected(self) -> None:
        """A bogus unit (``century``) must raise ValueError."""
        fs = self._resample_fs("value__resample_1_century_mean", ["region"])
        with pytest.raises(ValueError, match=r"(?i)unit|unsupported|century"):
            self.implementation_class().calculate_feature(self.test_data, fs)

    def test_bad_agg_median_rejected(self) -> None:
        """``median`` is not in the v1 agg set and must raise ValueError."""
        fs = self._resample_fs("value__resample_1_hour_median", ["region"])
        with pytest.raises(ValueError, match=r"(?i)agg|unsupported|median"):
            self.implementation_class().calculate_feature(self.test_data, fs)

    def test_bad_agg_last_rejected(self) -> None:
        """Order-DEPENDENT ``last`` is deliberately excluded in v1 and must raise ValueError."""
        fs = self._resample_fs("value__resample_1_hour_last", ["region"])
        with pytest.raises(ValueError, match=r"(?i)agg|unsupported|last"):
            self.implementation_class().calculate_feature(self.test_data, fs)

    def test_n_zero_rejected(self) -> None:
        """``n=0`` is not a valid bucket size and must raise ValueError."""
        fs = self._resample_fs("value__resample_0_hour_mean", ["region"])
        with pytest.raises(ValueError, match=r"(?i)positive|n|0"):
            self.implementation_class().calculate_feature(self.test_data, fs)

    def test_missing_time_column_rejected(self) -> None:
        """A missing ``time_column`` must raise a clear ValueError naming it."""
        table = pa.table(
            {
                "id": pa.array(_RESAMPLE_IDS, type=pa.int64()),
                "region": pa.array(_RESAMPLE_REGIONS, type=pa.string()),
                "value": pa.array(_RESAMPLE_VALUES, type=pa.float64()),
            }
        )
        data = self.create_test_data(table)
        fs = self._resample_fs("value__resample_1_hour_mean", ["region"])
        with pytest.raises(ValueError, match=r"(?i)ts|time_column|missing|column"):
            self.implementation_class().calculate_feature(data, fs)

    def test_missing_source_column_rejected(self) -> None:
        """A missing source value column must raise a clear ValueError naming it."""
        table = pa.table(
            {
                "id": pa.array(_RESAMPLE_IDS, type=pa.int64()),
                "region": pa.array(_RESAMPLE_REGIONS, type=pa.string()),
                "ts": pa.array(_RESAMPLE_TIMESTAMPS, type=pa.timestamp("us", tz="UTC")),
            }
        )
        data = self.create_test_data(table)
        fs = self._resample_fs("value__resample_1_hour_mean", ["region"])
        with pytest.raises(ValueError, match=r"(?i)value|missing|column"):
            self.implementation_class().calculate_feature(data, fs)

    def test_multi_column_in_features_rejected(self) -> None:
        """calculate_feature must reject features with multiple in_features (MAX_IN_FEATURES=1)."""
        feature = Feature(
            "bad_multi_col",
            options=Options(
                context={
                    "in_features": ["value", "other_value"],
                    "time_column": "ts",
                    "partition_by": ["region"],
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        with pytest.raises(ValueError, match=r"(?i)at most 1|in_features|single"):
            self.implementation_class().calculate_feature(self.test_data, fs)
