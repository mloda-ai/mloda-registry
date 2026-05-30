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

Expected-value literals were computed from PyArrow as the offline oracle
(``pyarrow.compute.floor_temporal``, ``ceil_temporal``, ``round_temporal``
with ``week_starts_monday=True`` and ``ceil_is_strictly_greater=False``)
and pasted in. PyArrow is the reference; this paste-in approach pins the
spec independent of the production implementation.

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

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.helpers import extract_column as _extract_column
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda.user import Feature


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


def _create_bucket_arrow_table() -> pa.Table:
    """Create the 8-row PyArrow table used by every time-bucketization test."""
    return pa.table(
        {
            "timestamp": pa.array(
                _BUCKET_TIMESTAMPS,
                type=pa.timestamp("us", tz="UTC"),
            ),
        }
    )


# ---------------------------------------------------------------------------
# Expected values (offline-computed from PyArrow as the oracle)
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
# PyArrow's ``round_temporal`` (default tie-break) produces 14:30 here, so
# that's the spec we pin. Other backends must agree.
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


class TimeBucketizationTestBase(DataOpsTestBase):
    """Abstract base class for time-bucketization framework tests.

    Subclasses combine this with a framework mixin (``PyArrowTestMixin``,
    ``PandasTestMixin``, etc.) and a one-liner ``implementation_class``
    classmethod returning the framework-specific feature group.

    All five backends in v1 support every (op, unit) combination for n=1,
    plus ``n>1`` for the fixed-freq units (minute / hour / day). The
    ``supported_ops`` classmethod reports the three op subtypes; framework
    test classes may override to a subset.
    """

    ALL_OPS = {"floor", "ceil", "round"}

    @classmethod
    def supported_ops(cls) -> set[str]:
        """Override to restrict supported ops for a framework (none today)."""
        return cls.ALL_OPS

    @classmethod
    def reference_implementation_class(cls) -> Any:
        # Lazy import so the testing module imports cleanly even before the
        # production code exists (Red phase, matrix drift check).
        from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.pyarrow_time_bucketization import (  # noqa: E501
            PyArrowTimeBucketization,
        )

        return PyArrowTimeBucketization

    # -- Setup: use the dedicated 8-row bucketization fixture ---------------

    def setup_method(self) -> None:
        """Override the canonical-fixture setup to use the dedicated 8-row table."""
        super().setup_method()  # connections + canonical data (mostly unused)
        self._arrow_table = _create_bucket_arrow_table()
        self.test_data = self.create_test_data(self._arrow_table)

    # -- Per-op / per-unit pinned-value tests --------------------------------

    def _assert_equal_with_nulls(self, actual: list[Any], expected: list[Any]) -> None:
        assert len(actual) == len(expected), f"row count {len(actual)} != expected {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            if e is None:
                assert a is None, f"row {i}: expected None, got {a!r}"
            else:
                # Accept ISO strings from SQLite (TEXT-stored timestamps).
                if isinstance(a, str):
                    a = datetime.fromisoformat(a)
                assert a == e, f"row {i}: {a!r} != {e!r}"

    def test_floor_5_minute(self) -> None:
        """``floor`` to 5-minute buckets. Row 2 (14:37:23) floors to 14:35:00."""
        fs = make_feature_set("timestamp__floor_5_minute")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert isinstance(result, self.get_expected_type())
        col = self.extract_column(result, "timestamp__floor_5_minute")
        self._assert_equal_with_nulls(col, EXPECTED_FLOOR_5_MINUTE)

    def test_floor_1_day(self) -> None:
        fs = make_feature_set("timestamp__floor_1_day")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "timestamp__floor_1_day")
        self._assert_equal_with_nulls(col, EXPECTED_FLOOR_1_DAY)

    def test_floor_1_week(self) -> None:
        """ISO-Monday-anchored week floor. Sun 2023-01-01 -> Mon 2022-12-26."""
        fs = make_feature_set("timestamp__floor_1_week")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "timestamp__floor_1_week")
        self._assert_equal_with_nulls(col, EXPECTED_FLOOR_1_WEEK)
        # Explicit pin on the most surprising row.
        assert col[0] == datetime(2022, 12, 26, 0, 0, 0, tzinfo=timezone.utc)

    def test_floor_1_month(self) -> None:
        fs = make_feature_set("timestamp__floor_1_month")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "timestamp__floor_1_month")
        self._assert_equal_with_nulls(col, EXPECTED_FLOOR_1_MONTH)

    def test_floor_1_year(self) -> None:
        fs = make_feature_set("timestamp__floor_1_year")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "timestamp__floor_1_year")
        self._assert_equal_with_nulls(col, EXPECTED_FLOOR_1_YEAR)

    def test_ceil_5_minute(self) -> None:
        fs = make_feature_set("timestamp__ceil_5_minute")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "timestamp__ceil_5_minute")
        self._assert_equal_with_nulls(col, EXPECTED_CEIL_5_MINUTE)

    def test_ceil_1_day_idempotent_on_aligned(self) -> None:
        """``ceil(aligned, 1_day) == aligned`` for rows 0, 1, 5 (all midnight)."""
        fs = make_feature_set("timestamp__ceil_1_day")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "timestamp__ceil_1_day")
        # Row 0 and row 1 are day-aligned; ceil must not advance.
        assert col[0] == datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        assert col[1] == datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc)

    def test_ceil_1_day_advances_on_non_aligned(self) -> None:
        """Row 2 (14:37:23) ceils up to next day midnight."""
        fs = make_feature_set("timestamp__ceil_1_day")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "timestamp__ceil_1_day")
        assert col[2] == datetime(2023, 6, 16, 0, 0, 0, tzinfo=timezone.utc)
        # And full list:
        self._assert_equal_with_nulls(col, EXPECTED_CEIL_1_DAY)

    def test_ceil_1_year_crosses_year_boundary(self) -> None:
        """Row 5 (2023-12-31 23:59:59) ceils up to 2024-01-01."""
        fs = make_feature_set("timestamp__ceil_1_year")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "timestamp__ceil_1_year")
        assert col[5] == datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        self._assert_equal_with_nulls(col, EXPECTED_CEIL_1_YEAR)

    def test_round_5_minute(self) -> None:
        """Sanity check round at 5-min granularity (no actual midpoint in fixture)."""
        fs = make_feature_set("timestamp__round_5_minute")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "timestamp__round_5_minute")
        self._assert_equal_with_nulls(col, EXPECTED_ROUND_5_MINUTE)

    def test_round_10_minute_at_midpoint(self) -> None:
        """Row 4 (14:25:00) is exactly midpoint between 10-min buckets 14:20 and 14:30.

        PyArrow's ``round_temporal`` default tie-break produces 14:30; this
        test pins that exact value as the cross-framework spec. Other backends
        must match PyArrow here.
        """
        fs = make_feature_set("timestamp__round_10_minute")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "timestamp__round_10_minute")
        assert col[4] == datetime(2023, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
        self._assert_equal_with_nulls(col, EXPECTED_ROUND_10_MINUTE)

    def test_round_1_day(self) -> None:
        fs = make_feature_set("timestamp__round_1_day")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "timestamp__round_1_day")
        self._assert_equal_with_nulls(col, EXPECTED_ROUND_1_DAY)

    # -- Cross-cutting semantics --------------------------------------------

    def test_week_starts_monday(self) -> None:
        """``floor_1_week`` is ISO-Monday-anchored.

        Sun 2023-01-01 -> Mon 2022-12-26 (previous Monday).
        Mon 2023-01-02 -> Mon 2023-01-02 (same Monday).
        """
        fs = make_feature_set("timestamp__floor_1_week")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "timestamp__floor_1_week")
        assert col[0] == datetime(2022, 12, 26, 0, 0, 0, tzinfo=timezone.utc)
        assert col[1] == datetime(2023, 1, 2, 0, 0, 0, tzinfo=timezone.utc)

    def test_null_timestamp_propagates(self) -> None:
        """Row 7 (null) must produce null output for floor, ceil, and round."""
        for op in ("floor_1_day", "ceil_1_day", "round_1_day"):
            fs = make_feature_set(f"timestamp__{op}")
            result = self.implementation_class().calculate_feature(self.test_data, fs)
            col = self.extract_column(result, f"timestamp__{op}")
            assert col[7] is None, f"row 7 must be None for op {op}, got {col[7]!r}"

    def test_timezone_preserved(self) -> None:
        """Output rows must retain a UTC tzinfo. Use a tz-agnostic equality check."""
        fs = make_feature_set("timestamp__floor_1_day")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "timestamp__floor_1_day")
        # Find the first non-null row; the canonical 8-row fixture has 7 non-null.
        non_null = [v for v in col if v is not None]
        assert non_null, "expected at least one non-null result row"
        sample = non_null[0]
        assert sample.tzinfo is not None, f"expected tz-aware datetime, got naive {sample!r}"
        assert sample.utcoffset() == timedelta(0), f"expected UTC offset 0 (input was UTC), got {sample.utcoffset()!r}"

    def test_output_rows_equal_input_rows(self) -> None:
        fs = make_feature_set("timestamp__floor_1_day")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert self.get_row_count(result) == 8

    def test_new_column_added(self) -> None:
        fs = make_feature_set("timestamp__floor_1_day")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "timestamp__floor_1_day")
        assert len(col) == 8

    def test_result_has_correct_type(self) -> None:
        fs = make_feature_set("timestamp__floor_1_day")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert isinstance(result, self.get_expected_type())

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
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "my_floored")
        self._assert_equal_with_nulls(col, EXPECTED_FLOOR_1_DAY)

    # -- Cross-framework comparison -----------------------------------------

    def _compare_bucket_with_reference(self, feature_name: str) -> None:
        """Run the feature on this framework and the PyArrow reference; assert equal."""
        fs = make_feature_set(feature_name)
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        ref = self.reference_implementation_class().calculate_feature(self._arrow_table, fs)

        result_col = self.extract_column(result, feature_name)
        ref_col = _extract_column(ref, feature_name)

        assert len(result_col) == len(ref_col), f"row count {len(result_col)} != reference {len(ref_col)}"
        for i, (ref_v, fw_v) in enumerate(zip(ref_col, result_col)):
            if ref_v is None:
                assert fw_v is None, f"row {i}: expected None, got {fw_v!r}"
            else:
                assert fw_v == ref_v, f"row {i}: {fw_v!r} != reference {ref_v!r}"

    def test_cross_framework_floor_1_day(self) -> None:
        self._compare_bucket_with_reference("timestamp__floor_1_day")

    def test_cross_framework_floor_2_day(self) -> None:
        """Multi-day fixed-freq floor must agree with PyArrow's epoch-anchored buckets.

        Backends that use a built-in ``time_bucket`` (DuckDB) anchor sub-month
        widths at 2000-01-03 by default, which diverges from PyArrow's
        epoch-anchored (multiples since 1970-01-01) buckets. This test pins
        agreement and would catch a regression on that anchor.
        """
        self._compare_bucket_with_reference("timestamp__floor_2_day")

    def test_cross_framework_floor_3_day(self) -> None:
        """3-day floor: extra coverage for multi-day anchor alignment with PyArrow."""
        self._compare_bucket_with_reference("timestamp__floor_3_day")

    def test_cross_framework_ceil_1_day(self) -> None:
        self._compare_bucket_with_reference("timestamp__ceil_1_day")

    def test_cross_framework_round_5_minute(self) -> None:
        self._compare_bucket_with_reference("timestamp__round_5_minute")

    def test_cross_framework_floor_1_week(self) -> None:
        self._compare_bucket_with_reference("timestamp__floor_1_week")

    def test_cross_framework_floor_1_month(self) -> None:
        self._compare_bucket_with_reference("timestamp__floor_1_month")

    def test_cross_framework_floor_1_year(self) -> None:
        self._compare_bucket_with_reference("timestamp__floor_1_year")

    # -- Error / validation --------------------------------------------------

    def test_unsupported_bucket_op_raises(self) -> None:
        """A bogus op in Options must raise ValueError at compute time."""
        feature = Feature(
            "bogus_result",
            options=Options(
                context={
                    "bucket_op": "bogus_1_day",
                    "in_features": "timestamp",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        with pytest.raises(ValueError, match=r"(?i)bucket_op|unsupported|could not extract"):
            self.implementation_class().calculate_feature(self.test_data, fs)

    def test_unsupported_unit_raises(self) -> None:
        """A bogus unit (e.g. ``century``) in Options must raise ValueError at compute."""
        feature = Feature(
            "bad_unit",
            options=Options(
                context={
                    "bucket_op": "floor_1_century",
                    "in_features": "timestamp",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        with pytest.raises(ValueError, match=r"(?i)unit|unsupported|century"):
            self.implementation_class().calculate_feature(self.test_data, fs)

    def test_n_zero_rejected(self) -> None:
        """``n=0`` is not a valid bucket size and must raise ValueError at compute."""
        feature = Feature(
            "zero_bucket",
            options=Options(
                context={
                    "bucket_op": "floor_0_day",
                    "in_features": "timestamp",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        with pytest.raises(ValueError, match=r"(?i)positive|n|0"):
            self.implementation_class().calculate_feature(self.test_data, fs)

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
            self.implementation_class().calculate_feature(self.test_data, fs)

    def test_non_timestamp_source_rejected(self) -> None:
        """A non-timestamp source column must be rejected with a clear ValueError."""
        # Build a one-column table where ``name`` is string-typed.
        string_table = pa.table({"name": pa.array(["a", "b", "c"], type=pa.string())})
        data = self.create_test_data(string_table)
        fs = make_feature_set("name__floor_1_day")
        import re

        with pytest.raises(ValueError, match=r"(?i)timestamp|datetime") as exc_info:
            self.implementation_class().calculate_feature(data, fs)
        # The error message should name the offending column.
        assert re.search(r"['\"]name['\"]", str(exc_info.value)) or "name" in str(exc_info.value), (
            f"Expected source column 'name' to be named in the error, got: {exc_info.value!r}"
        )

    def test_missing_source_column_raises_value_error(self) -> None:
        """All backends must raise ValueError (not KeyError or silent SQL error) when source col is absent.

        Today the behaviour diverges by backend:
        - DuckDB / SQLite: ``_assert_source_column_is_timestamp`` looks up a
          dtype map keyed on the column name; the missing key gives ``None``
          which fails the timestamp check silently (returns early), letting
          downstream SQL raise an opaque engine error.
        - Polars: ``data.collect_schema()[source_col]`` raises ``KeyError``.
        - Pandas: ``data[source_col]`` raises ``KeyError``.
        - PyArrow: ``data.column(source_col)`` raises ``KeyError`` /
          ``ArrowKeyError``.

        We want every backend to raise a clear ``ValueError`` naming the
        missing column.
        """
        other_table = pa.table({"not_timestamp": pa.array([1, 2, 3], type=pa.int64())})
        data = self.create_test_data(other_table)
        fs = make_feature_set("timestamp__floor_1_day")
        with pytest.raises(ValueError, match=r"(?i)timestamp|missing|column"):
            self.implementation_class().calculate_feature(data, fs)

    def test_multi_column_in_features_rejected_at_calculate(self) -> None:
        """calculate_feature must reject features with multiple in_features."""
        feature = Feature(
            "bad_multi_col",
            options=Options(
                context={
                    "bucket_op": "floor_1_day",
                    "in_features": ["timestamp", "other_ts"],
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        with pytest.raises(ValueError, match="at most 1"):
            self.implementation_class().calculate_feature(self.test_data, fs)
