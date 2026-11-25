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

Expected-value literals are hand-computed and pasted in and judge each backend. The kit
reference (``reference_resample``) is pinned by its own rows in ``pinning_rows.py`` (run by
``test_reference_pinning.py``), and the cross-framework test compares every backend against it. Raw pandas would differ on the
all-null ``sum`` bucket: pandas returns ``0.0`` there, the reference returns ``None``.

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

import math
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone, tzinfo
from typing import TYPE_CHECKING, Any, Callable
from zoneinfo import ZoneInfo

import pytest
from mloda.provider import FeatureSet
from mloda.user import DataType, Feature, Options

from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.helpers import (
    arrow_table_of,
    calculate_with,
    has_numeric_fixed_offset,
    make_feature_set,
    make_multi_feature_set,
)
from mloda.testing.feature_groups.data_operations.helpers import extract_column as _extract_column
from mloda.testing.feature_groups.data_operations.mixins.input_validation import (
    InputValidationCase,
    InputValidationTestMixin,
)
from mloda.testing.feature_groups.data_operations.mixins.nan_policy import NanPolicyTestMixin
from mloda.testing.feature_groups.data_operations.mixins.output_contract import OutputContractTestMixin
from mloda.testing.feature_groups.data_operations.mixins.pinned_rows import PinnedRowTestMixin
from mloda.testing.feature_groups.data_operations.mixins.result_type import ResultTypeTestMixin
from mloda.testing.feature_groups.data_operations.mixins.tz_aware import TzAwareSourceTestMixin
from mloda.testing.feature_groups.data_operations.pinning import PinningTable
from mloda.testing.feature_groups.data_operations.row_changing.resample.pinning_rows import RESAMPLE_PINNING
from mloda.testing.typed_data import TypedTestData

if TYPE_CHECKING:
    import pyarrow as pa

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


def _resample_columns() -> dict[str, tuple[DataType, list[Any]]]:
    """The four typed resample columns (id, region, ts, value)."""
    return {
        "id": (DataType.INT64, _RESAMPLE_IDS),
        "region": (DataType.STRING, _RESAMPLE_REGIONS),
        "ts": (DataType.TIMESTAMP_MICROS_TZ, _RESAMPLE_TIMESTAMPS),
        "value": (DataType.DOUBLE, _RESAMPLE_VALUES),
    }


def _create_resample_data(*, drop: str | None = None) -> TypedTestData:
    """Create the typed data used by every resample test, optionally without one column."""
    return TypedTestData({n: c for n, c in _resample_columns().items() if n != drop})


# Convenience bucket-start constructors.
def _h(hour: int, minute: int = 0) -> datetime:
    return datetime(2023, 1, 1, hour, minute, 0, tzinfo=_U)


# ---------------------------------------------------------------------------
# Expected values (hand-computed; keyed (region, bucket))
# ---------------------------------------------------------------------------
# The B hour09 bucket is non-empty (rows 5 & 7) but all-null: sum/mean=None,
# count=0. pandas would report sum=0.0 there; we pin None.

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
    ("B", _h(9)): None,  # all-null sum is None (pandas would say 0.0)
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

# NaN-policy resample buckets, evaluated on DataOpsTestBase.nan_policy_table() (grp/ts/val).
# 11-day buckets: one per grp. 1-day buckets: one per row, including three all-NaN
# buckets where the reference propagates NaN and still counts the row.
_BUCKET = datetime(2024, 1, 1, tzinfo=_U)


def _d(day: int) -> datetime:
    return datetime(2024, 1, day, tzinfo=_U)


_NAN = float("nan")

# sum/mean/min/max share one 1-day map: the reference value per bucket, with NaN on
# the three all-NaN buckets (A,01-02), (B,01-02), (B,01-04).
_NAN_1_DAY: dict[tuple[Any, ...], Any] = {
    ("A", _d(1)): 2.0,
    ("A", _d(2)): _NAN,
    ("A", _d(3)): 1.0,
    ("B", _d(1)): 1.0,
    ("B", _d(2)): _NAN,
    ("B", _d(3)): 3.0,
    ("B", _d(4)): _NAN,
}

# pandas' divergent counterpart: same as _NAN_1_DAY but None on the all-NaN buckets.
_NONE_1_DAY: dict[tuple[Any, ...], Any] = {
    ("A", _d(1)): 2.0,
    ("A", _d(2)): None,
    ("A", _d(3)): 1.0,
    ("B", _d(1)): 1.0,
    ("B", _d(2)): None,
    ("B", _d(3)): 3.0,
    ("B", _d(4)): None,
}

NAN_POLICY_RESAMPLE: dict[str, dict[tuple[Any, ...], Any]] = {
    "11_day_min": {("A", _BUCKET): 1.0, ("B", _BUCKET): 1.0},
    "11_day_max": {("A", _BUCKET): 2.0, ("B", _BUCKET): 3.0},
    "11_day_sum": {("A", _BUCKET): _NAN, ("B", _BUCKET): _NAN},
    "11_day_mean": {("A", _BUCKET): _NAN, ("B", _BUCKET): _NAN},
    "11_day_count": {("A", _BUCKET): 3, ("B", _BUCKET): 4},
    **{f"1_day_{agg}": _NAN_1_DAY for agg in ("sum", "mean", "min", "max")},
    "1_day_count": {
        ("A", _d(1)): 1,
        ("A", _d(2)): 1,
        ("A", _d(3)): 1,
        ("B", _d(1)): 1,
        ("B", _d(2)): 1,
        ("B", _d(3)): 1,
        ("B", _d(4)): 1,
    },
}

# Known per-backend divergence, pinned via nan_divergent_agg_types(): pandas skips NaN
# in sum/mean/count and returns null (count 0) for an all-NaN bucket. 11-day min/max
# match the policy, so they carry no entry here.
NAN_DIVERGENT_RESAMPLE: dict[str, dict[tuple[Any, ...], Any]] = {
    "11_day_sum": {("A", _BUCKET): 3.0, ("B", _BUCKET): 4.0},
    "11_day_mean": {("A", _BUCKET): 1.5, ("B", _BUCKET): 2.0},
    "11_day_count": {("A", _BUCKET): 2, ("B", _BUCKET): 2},
    **{f"1_day_{agg}": _NONE_1_DAY for agg in ("sum", "mean", "min", "max")},
    "1_day_count": {
        ("A", _d(1)): 1,
        ("A", _d(2)): 0,
        ("A", _d(3)): 1,
        ("B", _d(1)): 1,
        ("B", _d(2)): 0,
        ("B", _d(3)): 1,
        ("B", _d(4)): 0,
    },
}


# ---------------------------------------------------------------------------
# Reusable test base class
# ---------------------------------------------------------------------------


# Pandas and DuckDB sort buckets and partitions instead of keeping their first-occurrence order.
RESAMPLE_SORTED_BUCKET_DIVERGENCES: dict[str, dict[str, list[Any]]] = {
    "1-bucket-v__resample_1_hour_sum": {
        "ts": [datetime(2023, 1, 1, 7, 0, tzinfo=timezone.utc), datetime(2023, 1, 1, 8, 0, tzinfo=timezone.utc)],
        "v__resample_1_hour_sum": [2, 1],
    },
    "7-rows-v__resample_1_hour_sum": {
        "p": ["a", "a", "b"],
        "ts": [
            datetime(2023, 1, 1, 9, 0, tzinfo=timezone.utc),
            datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc),
            datetime(2023, 1, 1, 8, 0, tzinfo=timezone.utc),
        ],
        "v__resample_1_hour_sum": [4, 5, 6],
    },
    "24-partition_keys-v__resample_1_hour_sum": {"p": [1.0, math.nan], "v__resample_1_hour_sum": [2, 5]},
}


class ResampleTestBase(
    TzAwareSourceTestMixin,
    PinnedRowTestMixin,
    InputValidationTestMixin,
    OutputContractTestMixin,
    NanPolicyTestMixin,
    ResultTypeTestMixin,
    DataOpsTestBase,
):
    """Abstract base class for resample framework tests.

    Subclasses combine this with a framework mixin (``PyArrowTestMixin``,
    ``PandasTestMixin``, etc.) and a one-liner ``implementation_class``
    classmethod returning the framework-specific feature group.

    The expected base feature-group class is ``ResampleFeatureGroup`` and the
    backends are ``PyArrowResample``, ``PandasResample``,
    ``PolarsLazyResample`` and ``DuckdbResample``. SQLite is deferred to a
    later version and is intentionally not tested here.
    """

    @classmethod
    def pinning_table(cls) -> PinningTable:
        return RESAMPLE_PINNING

    @classmethod
    def pinned_divergences(cls) -> Mapping[str, Mapping[str, list[Any]]]:
        if not cls.framework_nan_is_null():
            return {}
        # The framework holds NaN as null: an all-NaN min or max is null, the NaN partition key reads back as None,
        # and NaN and None partition keys merge, collapsing two output buckets into one.
        return cls.served_pins(
            {
                "20-nan-v__resample_1_hour_min": {"v__resample_1_hour_min": [None]},
                "21-nan-v__resample_1_hour_max": {"v__resample_1_hour_max": [None]},
                "24-partition_keys-v__resample_1_hour_sum": {"p": [1.0, None], "v__resample_1_hour_sum": [2, 5]},
                "25-partition_keys-v__resample_1_hour_sum": {
                    "p": [None],
                    "ts": [datetime(2023, 1, 1, 8, 0, tzinfo=timezone.utc)],
                    "v__resample_1_hour_sum": [15],
                },
            }
        )

    @classmethod
    def nan_divergent_agg_types(cls) -> frozenset[str]:
        """The default plus ``min`` and ``max``, which a framework holding NaN as null returns as null for 1-day buckets."""
        divergent = super().nan_divergent_agg_types()
        return divergent | {"min", "max"} if cls.framework_nan_is_null() else divergent

    # -- OutputContractTestMixin configuration ----------------------------------

    def output_contract_feature_set(self) -> FeatureSet:
        return self._resample_fs("value__resample_1_hour_mean", ["region"])

    def output_contract_expected_row_count(self) -> int:
        # One row per region and hour bucket.
        return len(EXPECTED_1_HOUR_MEAN)

    @classmethod
    def reference_calculate(cls, table: pa.Table, features: Any) -> Any:
        """Run the kit reference function on the canonical table."""
        from mloda.testing.feature_groups.data_operations.row_changing.resample.reference import reference_resample

        return reference_resample(cls.feature_group_class(), table, features)

    # -- Setup: use the dedicated 12-row resample fixture -------------------

    @classmethod
    def source_data(cls) -> TypedTestData:
        return _create_resample_data()

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
        column: Callable[[Any, str], list[Any]] | None = None,
    ) -> dict[tuple[Any, ...], Any]:
        """Build a ``{(partition..., bucket_start): agg_value}`` map from a result.

        ``column`` extracts a named column as a Python list; defaults to
        ``self.extract_column`` so existing call sites are unaffected. Pass
        ``helpers.extract_column`` to build a map from a reference result.
        """
        column = column or self.extract_column
        bucket_col = [self._normalize_bucket_key(v) for v in column(result, time_column)]
        agg_col = column(result, feature_name)
        partition_cols = [column(result, p) for p in partition_by]

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
                assert got == pytest.approx(exp, rel=1e-6, nan_ok=True), f"bucket {key!r}: {got!r} != {exp!r}"
            else:
                assert got == exp, f"bucket {key!r}: {got!r} != {exp!r}"

    def _resample_fs(self, feature_name: str, partition_by: list[str]) -> FeatureSet:
        return make_feature_set(feature_name, partition_by=partition_by, time_column="ts")

    # -- ResultTypeTestMixin configuration --------------------------------------

    @classmethod
    def result_type_cases(cls) -> dict[str, FeatureSet]:
        return {
            agg: make_feature_set(f"value__resample_1_hour_{agg}", ["region"], time_column="ts")
            for agg in ("count", *cls._UNTYPED_RESAMPLE_AGGS)
        }

    _UNTYPED_RESAMPLE_AGGS = ("max", "mean", "min", "sum")

    @classmethod
    def untyped_result_type_ids(cls) -> frozenset[str]:
        return frozenset(cls._UNTYPED_RESAMPLE_AGGS)

    # -- NanPolicyTestMixin configuration ---------------------------------------

    @classmethod
    def nan_policy_cases(cls) -> dict[str, Any]:
        return NAN_POLICY_RESAMPLE

    @classmethod
    def nan_policy_divergent_cases(cls) -> dict[str, Any]:
        return NAN_DIVERGENT_RESAMPLE

    @classmethod
    def nan_policy_feature_name(cls, case: str) -> str:
        return f"val__resample_{case}"

    @classmethod
    def nan_policy_agg_type(cls, case: str) -> str:
        return case.rsplit("_", 1)[1]

    @classmethod
    def nan_policy_feature_set(cls, feature_name: str) -> FeatureSet:
        return make_feature_set(feature_name, partition_by=["grp"], time_column="ts")

    def nan_policy_skip_if_unsupported(self, case: str, agg_type: str, feature_name: str) -> None:
        """Resample skips nothing: every token in NAN_POLICY_RESAMPLE is always supported."""

    def nan_policy_extract_values(
        self,
        result: Any,
        feature_name: str,
        column: Callable[[Any, str], list[Any]],
    ) -> Any:
        bucket_map = self._build_resample_map(result, feature_name, "ts", ["grp"], column)
        row_count = len(column(result, feature_name))
        assert row_count == len(bucket_map), (
            f"{row_count} rows collapsed to {len(bucket_map)} map entries; duplicate (grp, bucket) rows?"
        )
        return bucket_map

    # -- Non-UTC zones --------------------------------------------------------

    # A sub-hour fixed offset is left out because Polars lazy cannot hold it; the named zone covers +05:30.
    @pytest.mark.parametrize(
        "tz",
        [timezone(timedelta(hours=5)), ZoneInfo("Asia/Kolkata")],
        ids=["fixed_plus_05", "named_asia_kolkata"],
    )
    def test_non_utc_zone_bucket(self, tz: tzinfo) -> None:
        """A non-UTC zone buckets on its own wall clock (UTC wall clock, source-zone labels, when the hook is True)."""
        rows = [datetime(2024, 1, 1, 2, 0, tzinfo=tz), datetime(2024, 1, 1, 7, 0, tzinfo=tz)]
        feature_name = "value__resample_1_day_count"
        fs = make_feature_set(feature_name, partition_by=[], time_column="ts")

        def typed(ts: list[datetime]) -> TypedTestData:
            return TypedTestData(
                {"ts": (DataType.TIMESTAMP_MICROS_TZ, ts), "value": (DataType.INT64, [10, 20])},
            )

        if self.rejects_numeric_offset_source() and has_numeric_fixed_offset(rows):
            with pytest.raises(ValueError, match=r"(?i)fixed.offset"):
                calculate_with(self.implementation_class(), self.create_test_data(typed(rows)), fs)
            return

        expected: dict[tuple[Any, ...], Any] = {(datetime(2024, 1, 1, tzinfo=tz),): 2}
        if self.tz_aware_source_reads_utc():
            utc_rows = [v.astimezone(timezone.utc) for v in rows]
            ref = self.reference_calculate(arrow_table_of(typed(utc_rows)), fs)
            utc_expected = self._build_resample_map(ref, feature_name, "ts", [], column=_extract_column)
            expected = {(*k[:-1], k[-1].astimezone(tz)): v for k, v in utc_expected.items()}
        result = calculate_with(self.implementation_class(), self.create_test_data(typed(rows)), fs)
        actual = self._build_resample_map(result, feature_name, "ts", [])
        self._assert_map_equals(actual, expected)
        assert {k[-1].utcoffset() for k in actual} == {k[-1].utcoffset() for k in expected}

    # -- Core per-partition value tests -------------------------------------

    def test_1_hour_mean_per_partition(self) -> None:
        """1-hour mean per region. Output row count == number of non-empty buckets (6)."""
        feature_name = "value__resample_1_hour_mean"
        fs = self._resample_fs(feature_name, ["region"])
        result = calculate_with(self.implementation_class(), self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        # Row-CHANGING assertion: one row per non-empty (region, hour) bucket.
        assert self.get_row_count(result) == len(EXPECTED_1_HOUR_MEAN)

        result_map = self._build_resample_map(result, feature_name, "ts", ["region"])
        self._assert_map_equals(result_map, EXPECTED_1_HOUR_MEAN, use_approx=True)

    def test_1_hour_sum_per_partition(self) -> None:
        feature_name = "value__resample_1_hour_sum"
        fs = self._resample_fs(feature_name, ["region"])
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        result_map = self._build_resample_map(result, feature_name, "ts", ["region"])
        self._assert_map_equals(result_map, EXPECTED_1_HOUR_SUM, use_approx=True)

    def test_1_hour_count_per_partition(self) -> None:
        feature_name = "value__resample_1_hour_count"
        fs = self._resample_fs(feature_name, ["region"])
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        result_map = self._build_resample_map(result, feature_name, "ts", ["region"])
        self._assert_map_equals(result_map, EXPECTED_1_HOUR_COUNT)

    def test_15_minute_mean_per_partition(self) -> None:
        """n>1 / sub-hour unit: guards epoch-anchored bucket alignment (:00/:15/:30/:45)."""
        feature_name = "value__resample_15_minute_mean"
        fs = self._resample_fs(feature_name, ["region"])
        result = calculate_with(self.implementation_class(), self.test_data, fs)

        assert self.get_row_count(result) == len(EXPECTED_15_MINUTE_MEAN)
        result_map = self._build_resample_map(result, feature_name, "ts", ["region"])
        self._assert_map_equals(result_map, EXPECTED_15_MINUTE_MEAN, use_approx=True)

    def test_whole_table_1_hour_mean(self) -> None:
        """No partition (partition_by=[]): one row per hour bucket across the whole table."""
        feature_name = "value__resample_1_hour_mean"
        fs = self._resample_fs(feature_name, [])
        result = calculate_with(self.implementation_class(), self.test_data, fs)

        assert self.get_row_count(result) == len(EXPECTED_1_HOUR_MEAN_WHOLE)
        result_map = self._build_resample_map(result, feature_name, "ts", [])
        self._assert_map_equals(result_map, EXPECTED_1_HOUR_MEAN_WHOLE, use_approx=True)

    # -- Row-changing semantics ---------------------------------------------

    def test_row_count_changes(self) -> None:
        """Output rows == number of non-empty buckets AND differs from input rows (12)."""
        feature_name = "value__resample_1_hour_mean"
        fs = self._resample_fs(feature_name, ["region"])
        result = calculate_with(self.implementation_class(), self.test_data, fs)

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
        result = calculate_with(self.implementation_class(), self.test_data, fs)
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
        result = calculate_with(self.implementation_class(), self.test_data, fs)
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
        mean_result = calculate_with(self.implementation_class(), self.test_data, mean_fs)
        mean_map = self._build_resample_map(mean_result, "value__resample_1_hour_mean", "ts", ["region"])
        assert mean_map[("A", _h(8))] == pytest.approx(7.0)

        sum_fs = self._resample_fs("value__resample_1_hour_sum", ["region"])
        sum_result = calculate_with(self.implementation_class(), self.test_data, sum_fs)
        sum_map = self._build_resample_map(sum_result, "value__resample_1_hour_sum", "ts", ["region"])
        assert sum_map[("A", _h(8))] == pytest.approx(14.0)

    def test_count_reflects_non_null(self) -> None:
        """count counts NON-NULL values: ('A', hour08) = 2 (the null is not counted)."""
        fs = self._resample_fs("value__resample_1_hour_count", ["region"])
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        count_map = self._build_resample_map(result, "value__resample_1_hour_count", "ts", ["region"])
        assert count_map[("A", _h(8))] == 2

    def test_all_null_bucket_emits_with_count_zero(self) -> None:
        """An all-null but non-empty bucket emits a row: count=0, mean/sum=None.

        ('B', hour09) has rows 5 & 7, both null. It is non-empty, so it must be
        present in the output with count 0 and mean / sum None.
        """
        count_fs = self._resample_fs("value__resample_1_hour_count", ["region"])
        count_result = calculate_with(self.implementation_class(), self.test_data, count_fs)
        count_map = self._build_resample_map(count_result, "value__resample_1_hour_count", "ts", ["region"])
        assert ("B", _h(9)) in count_map, "all-null bucket must still emit a row"
        assert count_map[("B", _h(9))] == 0

        mean_fs = self._resample_fs("value__resample_1_hour_mean", ["region"])
        mean_result = calculate_with(self.implementation_class(), self.test_data, mean_fs)
        mean_map = self._build_resample_map(mean_result, "value__resample_1_hour_mean", "ts", ["region"])
        assert mean_map[("B", _h(9))] is None

        sum_fs = self._resample_fs("value__resample_1_hour_sum", ["region"])
        sum_result = calculate_with(self.implementation_class(), self.test_data, sum_fs)
        sum_map = self._build_resample_map(sum_result, "value__resample_1_hour_sum", "ts", ["region"])
        assert sum_map[("B", _h(9))] is None

    def test_min_max_skip_nulls(self) -> None:
        """min / max skip nulls; ('A', hour08) over [10.0, None, 4.0] -> min 4.0, max 10.0."""
        min_fs = self._resample_fs("value__resample_1_hour_min", ["region"])
        min_result = calculate_with(self.implementation_class(), self.test_data, min_fs)
        min_map = self._build_resample_map(min_result, "value__resample_1_hour_min", "ts", ["region"])
        assert min_map[("A", _h(8))] == pytest.approx(4.0)

        max_fs = self._resample_fs("value__resample_1_hour_max", ["region"])
        max_result = calculate_with(self.implementation_class(), self.test_data, max_fs)
        max_map = self._build_resample_map(max_result, "value__resample_1_hour_max", "ts", ["region"])
        assert max_map[("A", _h(8))] == pytest.approx(10.0)

    # -- New column / type ---------------------------------------------------

    def test_new_column_named_exactly(self) -> None:
        """The aggregated column must be named exactly ``{col}__resample_{n}_{unit}_{agg}``."""
        feature_name = "value__resample_1_hour_mean"
        fs = self._resample_fs(feature_name, ["region"])
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        col = self.extract_column(result, feature_name)
        assert len(col) == len(EXPECTED_1_HOUR_MEAN)

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
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        result_map = self._build_resample_map(result, "value__resample_1_hour_mean", "ts", ["region"])
        self._assert_map_equals(result_map, EXPECTED_1_HOUR_MEAN, use_approx=True)

    # -- Several features in one FeatureSet ----------------------------------

    @staticmethod
    def _multi_feature_set(feature_names: list[str], partition_by: list[str] | list[list[str]]) -> FeatureSet:
        """Build a FeatureSet; a list of lists gives each feature its own partition_by."""
        specs: list[tuple[str, dict[str, Any]]] = []
        for i, name in enumerate(feature_names):
            part = partition_by[i] if partition_by and isinstance(partition_by[0], list) else partition_by
            specs.append((name, {"time_column": "ts", "partition_by": part}))
        return make_multi_feature_set(specs)

    def test_multiple_features_same_grid(self) -> None:
        """Features sharing a bucket grid are aggregated together from the original input, including a time-column source."""
        names = ["value__resample_1_hour_mean", "value__resample_1_hour_sum", "value__resample_1_hour_count"]
        ts_name = "ts__resample_1_hour_max"
        fs = self._multi_feature_set([*names, ts_name], ["region"])
        result = calculate_with(self.implementation_class(), self.test_data, fs)

        assert self.get_row_count(result) == len(EXPECTED_1_HOUR_MEAN)
        expected = [
            (names[0], EXPECTED_1_HOUR_MEAN, True),
            (names[1], EXPECTED_1_HOUR_SUM, True),
            (names[2], EXPECTED_1_HOUR_COUNT, False),
        ]
        for name, exp, approx in expected:
            result_map = self._build_resample_map(result, name, "ts", ["region"])
            self._assert_map_equals(result_map, exp, use_approx=approx)
        # A time-column source aggregates its own bucket start and stays aligned with the value sources.
        ts_map = self._build_resample_map(result, ts_name, "ts", ["region"])
        ts_map = {k: self._normalize_bucket_key(v) for k, v in ts_map.items()}
        self._assert_map_equals(ts_map, {k: k[-1] for k in EXPECTED_1_HOUR_MEAN})

    def test_empty_feature_set_returns_input(self) -> None:
        """A FeatureSet without features returns the input unchanged and never calls compute_resample."""

        def fail(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("compute_resample must not run for an empty FeatureSet")

        result = self.feature_group_class().resample(self.test_data, FeatureSet(), compute_resample=fail)
        assert result is self.test_data

    @pytest.mark.parametrize(
        "names, partition_by",
        [
            (["value__resample_1_hour_mean", "value__resample_15_minute_mean"], [["region"], ["region"]]),
            (["value__resample_1_hour_mean", "value__resample_1_hour_sum"], [["region"], []]),
        ],
        ids=["bucket_size", "partition_by"],
    )
    def test_mixed_bucket_grids_rejected(self, names: list[str], partition_by: list[list[str]]) -> None:
        """Features with different bucket grids in one FeatureSet are rejected, naming the remedy."""
        fs = self._multi_feature_set(names, partition_by)
        with pytest.raises(ValueError, match=r"(?i)bucket") as exc_info:
            calculate_with(self.implementation_class(), self.test_data, fs)
        assert "run_all" in str(exc_info.value)

    # -- Cross-framework comparison (map-based vs the kit reference) ------------

    def _compare_resample_with_reference(
        self,
        feature_name: str,
        partition_by: list[str],
        *,
        use_approx: bool = False,
    ) -> None:
        """Run the feature on this framework and the kit reference; assert maps equal."""
        fs = self._resample_fs(feature_name, partition_by)
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        ref = self.reference_calculate(self._arrow_table, fs)

        result_map = self._build_resample_map(result, feature_name, "ts", partition_by)
        ref_map = self._build_resample_map(ref, feature_name, "ts", partition_by, _extract_column)

        self._assert_map_equals(result_map, ref_map, use_approx=use_approx)

    @pytest.mark.parametrize(
        ("feature_name", "partition_by", "use_approx"),
        [
            pytest.param("value__resample_1_hour_mean", ["region"], True, id="1_hour_mean"),
            pytest.param("value__resample_1_hour_sum", ["region"], True, id="1_hour_sum"),
            # count is an exact integer, so no approximate comparison.
            pytest.param("value__resample_1_hour_count", ["region"], False, id="1_hour_count"),
            pytest.param("value__resample_15_minute_mean", ["region"], True, id="15_minute_mean"),
            pytest.param("value__resample_1_hour_mean", [], True, id="whole_table"),
        ],
    )
    def test_cross_framework(self, feature_name: str, partition_by: list[str], use_approx: bool) -> None:
        """Every resample spec must agree with the kit reference."""
        self._compare_resample_with_reference(feature_name, partition_by, use_approx=use_approx)

    @pytest.mark.parametrize("partition_by", [["region"], []], ids=["partitioned", "whole_table"])
    def test_input_not_mutated(self, partition_by: list[str]) -> None:
        """Resampling leaves the input frame's columns and values unchanged."""
        before = self._snapshot(self.test_data)
        fs = self._resample_fs("value__resample_1_hour_mean", partition_by)
        calculate_with(self.implementation_class(), self.test_data, fs)
        self._assert_snapshot_unchanged(before, self.test_data)

    def test_signed_zero_partition_keys(self) -> None:
        """Partition keys 0.0 and -0.0: the reference and most backends merge them into one group."""
        ts = datetime(2023, 1, 1, 8, 5, tzinfo=timezone.utc)
        table = TypedTestData(
            {
                "k": (DataType.DOUBLE, [0.0, -0.0, -0.0]),
                "ts": (DataType.TIMESTAMP_MICROS_TZ, [ts, ts, ts]),
                "v": (DataType.INT64, [1, 2, 3]),
            }
        )
        feature_name = "v__resample_1_hour_count"
        fs = make_feature_set(feature_name, partition_by=["k"], time_column="ts")

        result = calculate_with(self.implementation_class(), self.create_test_data(table), fs)
        result_counts = sorted(self.extract_column(result, feature_name))
        row_count = self.get_row_count(result)
        if self.merges_signed_zero_keys():
            assert row_count == 1, f"expected one merged group, got {row_count} rows"
            assert result_counts == [3]
        else:
            assert row_count == 2, f"expected two split groups, got {row_count} rows"
            assert result_counts == [1, 2]
            signs = sorted(math.copysign(1, v) for v in self.extract_column(result, "k"))
            assert signs == [-1.0, 1.0]

    # -- Error / validation --------------------------------------------------

    @pytest.mark.parametrize(
        ("feature_name", "match"),
        [
            pytest.param("value__resample_1_century_mean", r"(?i)unit|unsupported|century", id="bad_unit"),
            # ``median`` is not in the v1 agg set.
            pytest.param("value__resample_1_hour_median", r"(?i)agg|unsupported|median", id="bad_agg_median"),
            # Order-DEPENDENT ``last`` is deliberately excluded in v1.
            pytest.param("value__resample_1_hour_last", r"(?i)agg|unsupported|last", id="bad_agg_last"),
            # ``n=0`` is not a valid bucket size.
            pytest.param("value__resample_0_hour_mean", r"(?i)positive|n|0", id="n_zero"),
        ],
    )
    def test_bad_spec_rejected(self, feature_name: str, match: str) -> None:
        fs = self._resample_fs(feature_name, ["region"])
        with pytest.raises(ValueError, match=match):
            calculate_with(self.implementation_class(), self.test_data, fs)

    # -- InputValidationTestMixin configuration ---------------------------------

    @classmethod
    def input_validation_cases(cls) -> dict[str, Any]:
        return {
            "multi_column_in_features": InputValidationCase(
                "bad_multi_col",
                {"in_features": ["value", "other_value"], "time_column": "ts", "partition_by": ["region"]},
                "at most 1",
            ),
            "missing_source_column": InputValidationCase(
                "value__resample_1_hour_mean",
                {"partition_by": ["region"], "time_column": "ts"},
                r"Source column 'value' is not present",
                data=_create_resample_data(drop="value"),
            ),
            "empty_partition_by": None,
            "missing_partition_by_column": InputValidationCase(
                "value__resample_1_hour_mean",
                {"partition_by": ["no_such_col"], "time_column": "ts"},
                "partition_by 'no_such_col' is not present",
            ),
            # The resample time_column is its order key; the table omits it.
            "missing_order_by_column": InputValidationCase(
                "value__resample_1_hour_mean",
                {"partition_by": ["region"], "time_column": "ts"},
                "time_column 'ts' is not present",
                data=_create_resample_data(drop="ts"),
            ),
        }
