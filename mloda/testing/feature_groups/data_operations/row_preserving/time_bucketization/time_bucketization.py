"""Shared test base class, dedicated fixture, and expected values for time bucketization tests.

Each test verifies that a timestamp column is bucketed to a coarser interval
via floor / ceil / round to one of: minute, hour, day, week, month, year.

The canonical 12-row dataset in ``DataOperationsTestDataCreator`` is unusable
here because every timestamp is midnight UTC on a January 2023 day. Floor /
ceil / round to ``day`` would be a no-op and tell us nothing about the
bucketization logic. Instead this module ships its own 8-row UTC timestamp
fixture (``_BUCKET_TIMESTAMPS``) that exercises sub-day, sub-hour, and
sub-minute buckets, week-start alignment (ISO Monday), year/leap-day
boundary crossings, and null propagation.

Expected-value literals are pasted in, so they pin the spec independent of
any implementation; the pure Python reference computes the cross-framework
comparison values.

Fixture row layout::

    Row | Timestamp (UTC)            | Purpose
    ----+----------------------------+------------------------------------
    0   | 2023-01-01 00:00:00 (Sun)  | Day-aligned; ISO week floors to Mon Dec 26 2022
    1   | 2023-01-02 00:00:00 (Mon)  | Mon-aligned; floor_1_week = self
    2   | 2023-06-15 14:37:23 (Thu)  | Non-aligned across every unit
    3   | 2023-06-15 14:30:00 (Thu)  | 5/10/15-min aligned
    4   | 2023-06-15 14:25:00 (Thu)  | 5-min aligned; 10-min midpoint
    5   | 2023-12-31 23:59:59 (Sun)  | Ceil-to-year crosses to 2024-01-01
    6   | 2024-02-29 12:00:00 (Thu)  | Leap day edge case
    7   | None                        | Null propagation
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from mloda.provider import FeatureSet
from mloda.user import DataType, Feature, Options

from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.helpers import (
    assert_values_with_nulls,
    calculate_with,
    make_feature_set,
)
from mloda.testing.feature_groups.data_operations.helpers import extract_column as _extract_column
from mloda.testing.feature_groups.data_operations.mixins.input_validation import (
    InputValidationCase,
    InputValidationTestMixin,
)
from mloda.testing.feature_groups.data_operations.mixins.output_contract import OutputContractTestMixin
from mloda.testing.feature_groups.data_operations.mixins.result_type import ResultTypeTestMixin
from mloda.testing.feature_groups.data_operations.mixins.tz_aware import TzAwarePinningTestMixin
from mloda.testing.feature_groups.data_operations.pinning import PinningRow, PinningTable
from mloda.testing.feature_groups.data_operations.row_preserving.time_bucketization.pinning_rows import (
    TIME_BUCKETIZATION_PINNING,
)
from mloda.testing.typed_data import TypedTestData

# ---------------------------------------------------------------------------
# Dedicated 8-row fixture (UTC timestamps)
# ---------------------------------------------------------------------------

_BUCKET_TIMESTAMPS: list[datetime | None] = [
    datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),  # 0 Sun, day-aligned
    datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc),  # 1 Mon, week start
    datetime(2023, 6, 15, 14, 37, 23, tzinfo=timezone.utc),  # 2 Thu, non-aligned
    datetime(2023, 6, 15, 14, 30, 0, tzinfo=timezone.utc),  # 3 Thu, 5min-aligned
    datetime(2023, 6, 15, 14, 25, 0, tzinfo=timezone.utc),  # 4 Thu, 5min-aligned
    datetime(2023, 12, 31, 23, 59, 59, tzinfo=timezone.utc),  # 5 Sun, year edge
    datetime(2024, 2, 29, 12, 0, 0, tzinfo=timezone.utc),  # 6 Thu, leap day
    None,  # 7 null
]


def _create_bucket_data() -> TypedTestData:
    """Create the typed data used by every time-bucketization test."""
    return TypedTestData({"timestamp": (DataType.TIMESTAMP_MICROS_TZ, _BUCKET_TIMESTAMPS)})


# ---------------------------------------------------------------------------
# Expected values (pasted in)
# ---------------------------------------------------------------------------

EXPECTED_FLOOR_5_MINUTE: list[Any] = [
    datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 6, 15, 14, 35, 0, tzinfo=timezone.utc),
    datetime(2023, 6, 15, 14, 30, 0, tzinfo=timezone.utc),
    datetime(2023, 6, 15, 14, 25, 0, tzinfo=timezone.utc),
    datetime(2023, 12, 31, 23, 55, 0, tzinfo=timezone.utc),
    datetime(2024, 2, 29, 12, 0, 0, tzinfo=timezone.utc),
    None,
]

EXPECTED_FLOOR_1_DAY: list[Any] = [
    datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 6, 15, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 6, 15, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 6, 15, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 12, 31, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2024, 2, 29, 0, 0, 0, tzinfo=timezone.utc),
    None,
]

EXPECTED_FLOOR_1_WEEK: list[Any] = [
    datetime(2022, 12, 26, 0, 0, 0, tzinfo=timezone.utc),  # Sun -> previous Mon
    datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc),  # Mon -> same Mon
    datetime(2023, 6, 12, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 6, 12, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 6, 12, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 12, 25, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2024, 2, 26, 0, 0, 0, tzinfo=timezone.utc),
    None,
]

EXPECTED_FLOOR_1_MONTH: list[Any] = [
    datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 6, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 6, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 6, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 12, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2024, 2, 1, 0, 0, 0, tzinfo=timezone.utc),
    None,
]

EXPECTED_FLOOR_1_YEAR: list[Any] = [
    datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    None,
]

EXPECTED_CEIL_5_MINUTE: list[Any] = [
    datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 6, 15, 14, 40, 0, tzinfo=timezone.utc),
    datetime(2023, 6, 15, 14, 30, 0, tzinfo=timezone.utc),  # already 5-min aligned
    datetime(2023, 6, 15, 14, 25, 0, tzinfo=timezone.utc),  # already 5-min aligned
    datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2024, 2, 29, 12, 0, 0, tzinfo=timezone.utc),
    None,
]

EXPECTED_CEIL_1_DAY: list[Any] = [
    datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),  # already day-aligned
    datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc),  # already day-aligned
    datetime(2023, 6, 16, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 6, 16, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 6, 16, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2024, 3, 1, 0, 0, 0, tzinfo=timezone.utc),
    None,
]

EXPECTED_CEIL_1_YEAR: list[Any] = [
    datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    None,
]

EXPECTED_ROUND_5_MINUTE: list[Any] = [
    datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 6, 15, 14, 35, 0, tzinfo=timezone.utc),
    datetime(2023, 6, 15, 14, 30, 0, tzinfo=timezone.utc),
    datetime(2023, 6, 15, 14, 25, 0, tzinfo=timezone.utc),
    datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2024, 2, 29, 12, 0, 0, tzinfo=timezone.utc),
    None,
]

# Row 4 (14:25:00) is at the midpoint of the 10-minute buckets 14:20 / 14:30.
# Ties round up, so 14:30 is the spec we pin. Other backends must agree.
EXPECTED_ROUND_10_MINUTE: list[Any] = [
    datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 6, 15, 14, 40, 0, tzinfo=timezone.utc),
    datetime(2023, 6, 15, 14, 30, 0, tzinfo=timezone.utc),
    datetime(2023, 6, 15, 14, 30, 0, tzinfo=timezone.utc),  # midpoint -> 14:30
    datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2024, 2, 29, 12, 0, 0, tzinfo=timezone.utc),
    None,
]

EXPECTED_ROUND_1_DAY: list[Any] = [
    datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 6, 16, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 6, 16, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2023, 6, 16, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2024, 3, 1, 0, 0, 0, tzinfo=timezone.utc),
    None,
]


# ---------------------------------------------------------------------------
# Reusable test base class
# ---------------------------------------------------------------------------


class TimeBucketizationTestBase(
    TzAwarePinningTestMixin, InputValidationTestMixin, OutputContractTestMixin, ResultTypeTestMixin, DataOpsTestBase
):
    """Abstract base class for time-bucketization framework tests.

    Subclasses combine this with a framework mixin (``PyArrowTestMixin``,
    ``PandasTestMixin``, etc.) and a one-liner ``implementation_class``
    classmethod returning the framework-specific feature group.

    All five backends in v1 support every (op, unit) combination for n=1,
    plus ``n>1`` for the fixed-freq units (minute / hour / day). The
    ``supported_ops`` classmethod reports the three op subtypes; framework
    test classes may override to a subset.
    """

    @classmethod
    def pinning_table(cls) -> PinningTable:
        return TIME_BUCKETIZATION_PINNING

    @classmethod
    def rejects_non_utc_source(cls) -> bool:
        """True when a tz-aware source with a non-zero offset is rejected with a ValueError."""
        return False

    @classmethod
    def holds_sub_hour_offset(cls) -> bool:
        """False when the framework cannot hold a fixed offset that is not a whole number of hours."""
        return True

    @classmethod
    def holds_nanosecond_timestamps(cls) -> bool:
        """False when the framework cannot hold a nanosecond timestamp column."""
        return True

    @classmethod
    def buckets_tz_aware_nanosecond_source(cls) -> bool:
        """False when the framework rejects a tz-aware nanosecond source up front."""
        return True

    # -- OutputContractTestMixin configuration ----------------------------------

    def output_contract_feature_set(self) -> FeatureSet:
        return make_feature_set("timestamp__floor_1_day")

    ALL_OPS = {"floor", "ceil", "round"}

    # -- ResultTypeTestMixin configuration --------------------------------------

    @classmethod
    def result_type_cases(cls) -> dict[str, FeatureSet]:
        return {op: make_feature_set(f"timestamp__{op}_1_day") for op in sorted(cls.ALL_OPS)}

    @classmethod
    def supported_ops(cls) -> set[str]:
        """Subtypes the Implementation declares; the full set for a stub."""
        return cls.declared_supported(cls.ALL_OPS)

    @classmethod
    def reference_calculate(cls, table: Any, features: Any) -> Any:
        """Run the pure Python reference function on the canonical table."""
        from mloda.testing.feature_groups.data_operations.row_preserving.time_bucketization.reference import (
            reference_time_bucketization,
        )

        return reference_time_bucketization(cls.feature_group_class(), table, features)

    # -- Setup: use the dedicated 8-row bucketization fixture ---------------

    @classmethod
    def source_data(cls) -> TypedTestData:
        return _create_bucket_data()

    # -- Per-op / per-unit pinned-value tests --------------------------------

    def test_floor_5_minute(self) -> None:
        """``floor`` to 5-minute buckets. Row 2 (14:37:23) floors to 14:35:00."""
        fs = make_feature_set("timestamp__floor_5_minute")
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        assert isinstance(result, self.get_expected_type())
        col = self.extract_column(result, "timestamp__floor_5_minute")
        assert_values_with_nulls(col, EXPECTED_FLOOR_5_MINUTE, parse_iso=True)

    def test_floor_1_day(self) -> None:
        fs = make_feature_set("timestamp__floor_1_day")
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        col = self.extract_column(result, "timestamp__floor_1_day")
        assert_values_with_nulls(col, EXPECTED_FLOOR_1_DAY, parse_iso=True)

    def test_floor_1_week(self) -> None:
        """ISO-Monday-anchored week floor. Sun 2023-01-01 -> Mon 2022-12-26."""
        fs = make_feature_set("timestamp__floor_1_week")
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        col = self.extract_column(result, "timestamp__floor_1_week")
        assert_values_with_nulls(col, EXPECTED_FLOOR_1_WEEK, parse_iso=True)
        # Explicit pin on the most surprising row.
        assert col[0] == datetime(2022, 12, 26, 0, 0, 0, tzinfo=timezone.utc)

    def test_floor_1_month(self) -> None:
        fs = make_feature_set("timestamp__floor_1_month")
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        col = self.extract_column(result, "timestamp__floor_1_month")
        assert_values_with_nulls(col, EXPECTED_FLOOR_1_MONTH, parse_iso=True)

    def test_floor_1_year(self) -> None:
        fs = make_feature_set("timestamp__floor_1_year")
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        col = self.extract_column(result, "timestamp__floor_1_year")
        assert_values_with_nulls(col, EXPECTED_FLOOR_1_YEAR, parse_iso=True)

    def test_ceil_5_minute(self) -> None:
        fs = make_feature_set("timestamp__ceil_5_minute")
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        col = self.extract_column(result, "timestamp__ceil_5_minute")
        assert_values_with_nulls(col, EXPECTED_CEIL_5_MINUTE, parse_iso=True)

    def test_ceil_1_day_idempotent_on_aligned(self) -> None:
        """``ceil(aligned, 1_day) == aligned`` for rows 0, 1, 5 (all midnight)."""
        fs = make_feature_set("timestamp__ceil_1_day")
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        col = self.extract_column(result, "timestamp__ceil_1_day")
        # Row 0 and row 1 are day-aligned; ceil must not advance.
        assert col[0] == datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert col[1] == datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc)

    def test_ceil_1_day_advances_on_non_aligned(self) -> None:
        """Row 2 (14:37:23) ceils up to next day midnight."""
        fs = make_feature_set("timestamp__ceil_1_day")
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        col = self.extract_column(result, "timestamp__ceil_1_day")
        assert col[2] == datetime(2023, 6, 16, 0, 0, 0, tzinfo=timezone.utc)
        # And full list:
        assert_values_with_nulls(col, EXPECTED_CEIL_1_DAY, parse_iso=True)

    def test_ceil_1_year_crosses_year_boundary(self) -> None:
        """Row 5 (2023-12-31 23:59:59) ceils up to 2024-01-01."""
        fs = make_feature_set("timestamp__ceil_1_year")
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        col = self.extract_column(result, "timestamp__ceil_1_year")
        assert col[5] == datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert_values_with_nulls(col, EXPECTED_CEIL_1_YEAR, parse_iso=True)

    def test_round_5_minute(self) -> None:
        """Sanity check round at 5-min granularity (no actual midpoint in fixture)."""
        fs = make_feature_set("timestamp__round_5_minute")
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        col = self.extract_column(result, "timestamp__round_5_minute")
        assert_values_with_nulls(col, EXPECTED_ROUND_5_MINUTE, parse_iso=True)

    def test_round_10_minute_at_midpoint(self) -> None:
        """Row 4 (14:25:00) is exactly midpoint between 10-min buckets 14:20 and 14:30.

        Ties round up to 14:30; this test pins that value as the cross-framework
        spec. Every backend must match it.
        """
        fs = make_feature_set("timestamp__round_10_minute")
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        col = self.extract_column(result, "timestamp__round_10_minute")
        assert col[4] == datetime(2023, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
        assert_values_with_nulls(col, EXPECTED_ROUND_10_MINUTE, parse_iso=True)

    def test_round_1_day(self) -> None:
        fs = make_feature_set("timestamp__round_1_day")
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        col = self.extract_column(result, "timestamp__round_1_day")
        assert_values_with_nulls(col, EXPECTED_ROUND_1_DAY, parse_iso=True)

    # -- Pinned rows ---------------------------------------------------------

    @classmethod
    def checks_pinned_types(cls) -> bool:
        """Every pinned row keeps its declared type: the unit and tz-awareness; the zone is compared by value."""
        return True

    def rejection_match(self, row: PinningRow) -> str | None:
        """Sub-hour rows skip where unholdable, then the declared non-UTC and numeric-offset rejections."""
        column = next(iter(row.inputs))
        values = row.inputs[column][1]
        offsets = [v.utcoffset() for v in values if v is not None and v.utcoffset() is not None]
        if not self.holds_sub_hour_offset() and any(o.total_seconds() % 3600 for o in offsets):
            pytest.skip("framework cannot hold a sub-hour fixed offset")
        if self.rejects_non_utc_source() and any(o != timedelta(0) for o in offsets):
            return "non-UTC"
        return super().rejection_match(row)

    # -- Nanosecond range edge ---------------------------------------------------

    @pytest.mark.parametrize("tz", [None, "UTC"], ids=["naive", "utc"])
    @pytest.mark.parametrize(
        ("bucket_op", "value"),
        [
            pytest.param("ceil_1_year", datetime(2262, 3, 1), id="ceil_1_year_high"),
            pytest.param("floor_1_year", datetime(1677, 9, 22), id="floor_1_year_low"),
            pytest.param("floor_1_day", datetime(1677, 9, 21, 0, 13), id="floor_1_day_low"),
            pytest.param("ceil_1_day", datetime(2262, 4, 11, 23), id="ceil_1_day_high"),
            pytest.param("round_1_day", datetime(2262, 4, 11, 23), id="round_1_day_high"),
            pytest.param("ceil_1_year", datetime(1677, 9, 22), id="ceil_1_year_low"),
            pytest.param("ceil_1_month", datetime(1677, 9, 22), id="ceil_1_month_low"),
            pytest.param("ceil_1_day", datetime(1677, 9, 21, 1), id="ceil_1_day_low"),
            pytest.param("round_1_year", datetime(2262, 3, 1), id="round_1_year_high"),
            pytest.param("round_1_day", datetime(1677, 9, 21, 1), id="round_1_day_low"),
        ],
    )
    def test_nanosecond_boundary_out_of_range_rejected(self, tz: str | None, bucket_op: str, value: datetime) -> None:
        """A bucket boundary outside the nanosecond range raises a ValueError naming the column."""
        pa = pytest.importorskip("pyarrow")
        if not self.holds_nanosecond_timestamps():
            pytest.skip("framework cannot hold a nanosecond timestamp column")
        if tz is None:
            typed = TypedTestData({"ts": (DataType.TIMESTAMP_MICROS, [value, None])})
        else:
            typed = TypedTestData({"ts": (DataType.TIMESTAMP_MICROS_TZ, [value.replace(tzinfo=timezone.utc), None])})
        data = self.create_test_data(typed, arrow_types={"ts": pa.timestamp("ns", tz=tz)})
        fs = make_feature_set(f"ts__{bucket_op}")
        with pytest.raises(ValueError, match=r"['\"]ts['\"]"):
            calculate_with(self.implementation_class(), data, fs)

    @pytest.mark.parametrize(
        ("bucket_op", "value", "expected"),
        [
            pytest.param("floor_1_year", datetime(2262, 3, 1), datetime(2262, 1, 1), id="floor_1_year_high"),
            pytest.param("floor_1_hour", datetime(1677, 9, 21, 1, 30), datetime(1677, 9, 21, 1), id="floor_1_hour_low"),
            pytest.param("ceil_1_day", datetime(2262, 4, 10, 12), datetime(2262, 4, 11), id="ceil_1_day_high"),
            pytest.param(
                "round_1_hour", datetime(2262, 4, 10, 12, 20), datetime(2262, 4, 10, 12), id="round_1_hour_high"
            ),
            pytest.param("ceil_1_year", datetime(2261, 6, 1), datetime(2262, 1, 1), id="ceil_1_year_high"),
        ],
    )
    @pytest.mark.parametrize("tz", [None, "UTC"], ids=["naive", "utc"])
    def test_nanosecond_boundary_inside_range_computed(
        self, tz: str | None, bucket_op: str, value: datetime, expected: datetime
    ) -> None:
        """A boundary just inside the nanosecond range is still computed; a null row stays null."""
        pa = pytest.importorskip("pyarrow")
        if not self.holds_nanosecond_timestamps():
            pytest.skip("framework cannot hold a nanosecond timestamp column")
        if tz is not None and not self.buckets_tz_aware_nanosecond_source():
            pytest.skip("framework rejects a tz-aware nanosecond source")
        if tz is None:
            typed = TypedTestData({"ts": (DataType.TIMESTAMP_MICROS, [value, None])})
        else:
            typed = TypedTestData({"ts": (DataType.TIMESTAMP_MICROS_TZ, [value.replace(tzinfo=timezone.utc), None])})
            expected = expected.replace(tzinfo=timezone.utc)
        data = self.create_test_data(typed, arrow_types={"ts": pa.timestamp("ns", tz=tz)})
        fs = make_feature_set(f"ts__{bucket_op}")
        result = calculate_with(self.implementation_class(), data, fs)
        col = self.extract_column(result, f"ts__{bucket_op}")
        assert col[0] == expected
        assert col[1] is None

    # -- Cross-cutting semantics --------------------------------------------

    def test_week_starts_monday(self) -> None:
        """``floor_1_week`` is ISO-Monday-anchored.

        Sun 2023-01-01 -> Mon 2022-12-26 (previous Monday).
        Mon 2023-01-02 -> Mon 2023-01-02 (same Monday).
        """
        fs = make_feature_set("timestamp__floor_1_week")
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        col = self.extract_column(result, "timestamp__floor_1_week")
        assert col[0] == datetime(2022, 12, 26, 0, 0, 0, tzinfo=timezone.utc)
        assert col[1] == datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc)

    def test_null_timestamp_propagates(self) -> None:
        """Row 7 (null) must produce null output for floor, ceil, and round."""
        for op in ("floor_1_day", "ceil_1_day", "round_1_day"):
            fs = make_feature_set(f"timestamp__{op}")
            result = calculate_with(self.implementation_class(), self.test_data, fs)
            col = self.extract_column(result, f"timestamp__{op}")
            assert col[7] is None, f"row 7 must be None for op {op}, got {col[7]!r}"

    def test_timezone_preserved(self) -> None:
        """Output rows must retain a UTC tzinfo. Use a tz-agnostic equality check."""
        fs = make_feature_set("timestamp__floor_1_day")
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        col = self.extract_column(result, "timestamp__floor_1_day")
        # Find the first non-null row; the canonical 8-row fixture has 7 non-null.
        non_null = [v for v in col if v is not None]
        assert non_null, "expected at least one non-null result row"
        sample = non_null[0]
        assert sample.tzinfo is not None, f"expected tz-aware datetime, got naive {sample!r}"
        assert sample.utcoffset() == timedelta(0), f"expected UTC offset 0 (input was UTC), got {sample.utcoffset()!r}"

    def test_option_based_floor(self) -> None:
        """Option-based configuration (no string pattern) produces the same result."""
        feature = Feature(
            "my_floored",
            options=Options(
                context={
                    "bucket_op": "floor_1_day",
                    "in_features": "timestamp",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        col = self.extract_column(result, "my_floored")
        assert_values_with_nulls(col, EXPECTED_FLOOR_1_DAY, parse_iso=True)

    # -- Cross-framework comparison -----------------------------------------

    def _compare_bucket_with_reference(self, feature_name: str) -> None:
        """Run the feature on this framework and the pure Python reference; assert equal."""
        fs = make_feature_set(feature_name)
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        ref = self.reference_calculate(self._arrow_table, fs)

        result_col = self.extract_column(result, feature_name)
        ref_col = _extract_column(ref, feature_name)

        assert len(result_col) == len(ref_col), f"row count {len(result_col)} != reference {len(ref_col)}"
        for i, (ref_v, fw_v) in enumerate(zip(ref_col, result_col)):
            if ref_v is None:
                assert fw_v is None, f"row {i}: expected None, got {fw_v!r}"
            else:
                assert fw_v == ref_v, f"row {i}: {fw_v!r} != reference {ref_v!r}"

    @pytest.mark.parametrize(
        "feature_name",
        [
            pytest.param("timestamp__floor_1_day", id="floor_1_day"),
            # Backends with a built-in ``time_bucket`` (DuckDB) anchor sub-month widths
            # at 2000-01-03 by default, which diverges from the epoch-anchored
            # buckets (multiples since 1970-01-01). These two cases pin that anchor.
            pytest.param("timestamp__floor_2_day", id="floor_2_day"),
            pytest.param("timestamp__floor_3_day", id="floor_3_day"),
            pytest.param("timestamp__ceil_1_day", id="ceil_1_day"),
            pytest.param("timestamp__round_5_minute", id="round_5_minute"),
            pytest.param("timestamp__floor_6_hour", id="floor_6_hour"),
            pytest.param("timestamp__floor_1_week", id="floor_1_week"),
            pytest.param("timestamp__floor_1_month", id="floor_1_month"),
            pytest.param("timestamp__floor_1_year", id="floor_1_year"),
            pytest.param("timestamp__ceil_1_week", id="ceil_1_week"),
            pytest.param("timestamp__ceil_1_month", id="ceil_1_month"),
            pytest.param("timestamp__round_1_month", id="round_1_month"),
            pytest.param("timestamp__round_1_year", id="round_1_year"),
        ],
    )
    def test_cross_framework(self, feature_name: str) -> None:
        """Every bucket op must agree with the pure Python reference."""
        self._compare_bucket_with_reference(feature_name)

    # -- Error / validation --------------------------------------------------

    @pytest.mark.parametrize(
        ("result_name", "bucket_op", "match"),
        [
            pytest.param(
                "bogus_result",
                "bogus_1_day",
                r"(?i)bucket_op|unsupported|could not extract",
                id="unsupported_bucket_op",
            ),
            pytest.param("bad_unit", "floor_1_century", r"(?i)unit|unsupported|century", id="unsupported_unit"),
            # ``n=0`` is not a valid bucket size.
            pytest.param("zero_bucket", "floor_0_day", r"(?i)positive|n|0", id="n_zero"),
        ],
    )
    def test_invalid_bucket_op_rejected(self, result_name: str, bucket_op: str, match: str) -> None:
        """An invalid op in Options must raise ValueError at compute time."""
        feature = Feature(
            result_name,
            options=Options(
                context={
                    "bucket_op": bucket_op,
                    "in_features": "timestamp",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        with pytest.raises(ValueError, match=match):
            calculate_with(self.implementation_class(), self.test_data, fs)

    @pytest.mark.parametrize("unit", ["week", "month", "year"])
    def test_n_greater_than_one_for_calendar_unit_rejected(self, unit: str) -> None:
        """``n > 1`` for week / month / year is rejected in v1 with a clear error."""
        feature = Feature(
            "bad_multi",
            options=Options(
                context={
                    "bucket_op": f"floor_2_{unit}",
                    "in_features": "timestamp",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        with pytest.raises(ValueError, match=rf"(?i){unit}"):
            calculate_with(self.implementation_class(), self.test_data, fs)

    def test_non_timestamp_source_rejected(self) -> None:
        """A non-timestamp source column must be rejected with a clear ValueError."""
        # Build a one-column dataset where ``name`` is string-typed.
        data = self.create_test_data(TypedTestData({"name": (DataType.STRING, ["a", "b", "c"])}))
        fs = make_feature_set("name__floor_1_day")
        import re

        with pytest.raises(ValueError, match=r"(?i)timestamp|datetime") as exc_info:
            calculate_with(self.implementation_class(), data, fs)
        # The error message should name the offending column.
        assert re.search(r"['\"]name['\"]", str(exc_info.value)) or "name" in str(exc_info.value), (
            f"Expected source column 'name' to be named in the error, got: {exc_info.value!r}"
        )

    # -- InputValidationTestMixin configuration ---------------------------------

    @classmethod
    def input_validation_cases(cls) -> dict[str, Any]:
        missing_timestamp_data = TypedTestData({"not_timestamp": (DataType.INT64, [1, 2, 3])})
        return {
            "multi_column_in_features": InputValidationCase(
                "bad_multi_col",
                {"bucket_op": "floor_1_day", "in_features": ["timestamp", "other_ts"]},
                "at most 1",
            ),
            "missing_source_column": InputValidationCase(
                "timestamp__floor_1_day",
                {},
                "Source column 'timestamp' is not present",
                data=missing_timestamp_data,
            ),
            "empty_partition_by": None,
            "missing_partition_by_column": None,
            "missing_order_by_column": None,
        }
