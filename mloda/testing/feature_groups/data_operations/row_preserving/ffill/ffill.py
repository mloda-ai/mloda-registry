"""Shared test base, dedicated fixture, and expected values for ffill-by-time tests.

``ffill`` forward-fills a value column across time gaps. Within each partition,
rows are sorted by an ``order_by`` (time) column ascending, then the last
non-null value of the source column is carried FORWARD to fill nulls. The
operation is ROW-PRESERVING: the result has the same rows in the same original
order as the input, with one new ``{col}__ffill`` column appended.

Null rules:

- Leading nulls (before the first non-null in time order) stay NULL.
- A null that follows a non-null gets the carried value.
- Non-null source values pass through unchanged.

The canonical 12-row dataset in ``DataOperationsTestDataCreator`` is unusable
here because it has no partition column, no time gaps, and no nulls in a value
column. Instead this module ships its own dedicated 10-row UTC-timestamp
fixture that interleaves two partitions (``region`` A / B) in ROW order, so a
partition-aware fill differs from a naive whole-table fill, and orders the rows
so that TIME order differs from ROW order (so sorting actually matters).

Expected-value literals were computed offline with a pandas ``groupby`` /
``ffill`` snippet (per-partition spec) and pasted in. PyArrow is the reference
oracle; this paste-in approach pins the spec independent of the production
implementation. Note that ``pyarrow.compute.fill_null_forward`` does NOT
respect partition boundaries, so the production PyArrow implementation must
fill per partition; the reference comparison threads ``partition_by`` /
``order_by`` through both calls so the oracle fills per partition too.

Fixture row layout (``id`` is the passthrough row-order witness)::

    id | region | ts (UTC)              | value | Purpose
    ---+--------+----------------------+-------+--------------------------------
    0  | A      | 2023-01-01 10:00:00  | 1.0   | A: first non-null (by time: 2nd)
    1  | B      | 2023-01-01 10:30:00  | None  | B: leading null (still leading)
    2  | A      | 2023-01-01 11:00:00  | None  | A: interior null after 1.0 -> 1.0
    3  | B      | 2023-01-01 11:30:00  | 5.0   | B: first non-null
    4  | A      | 2023-01-01 09:00:00  | 0.5   | A: earliest by time (row order != time)
    5  | B      | 2023-01-01 13:00:00  | None  | B: trailing null -> 5.0
    6  | A      | 2023-01-01 13:30:00  | None  | A: trailing null -> 9.0
    7  | B      | 2023-01-01 12:00:00  | None  | B: interior null after 5.0 -> 5.0
    8  | A      | 2023-01-01 12:30:00  | 9.0   | A: non-null
    9  | B      | 2023-01-01 09:30:00  | None  | B: earliest by time -> leading null stays

    All 10 timestamps are GLOBALLY UNIQUE (no ties) so SQL backends that
    ``ORDER BY ts`` get a deterministic whole-table sort; per-partition relative
    time order is preserved (A: id4<id0<id2<id8<id6, B: id9<id1<id3<id7<id5).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.reserved_columns import ReservedColumnsTestMixin
from mloda.user import Feature

_U = timezone.utc


# ---------------------------------------------------------------------------
# Dedicated 10-row fixture (UTC timestamps, two interleaved partitions)
# ---------------------------------------------------------------------------

_FFILL_IDS: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

_FFILL_REGIONS: list[str] = ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]

_FFILL_TIMESTAMPS: list[datetime] = [
    datetime(2023, 1, 1, 10, 0, 0, tzinfo=_U),  # 0 A
    datetime(2023, 1, 1, 10, 30, 0, tzinfo=_U),  # 1 B
    datetime(2023, 1, 1, 11, 0, 0, tzinfo=_U),  # 2 A
    datetime(2023, 1, 1, 11, 30, 0, tzinfo=_U),  # 3 B
    datetime(2023, 1, 1, 9, 0, 0, tzinfo=_U),  # 4 A (earliest A by time)
    datetime(2023, 1, 1, 13, 0, 0, tzinfo=_U),  # 5 B
    datetime(2023, 1, 1, 13, 30, 0, tzinfo=_U),  # 6 A
    datetime(2023, 1, 1, 12, 0, 0, tzinfo=_U),  # 7 B
    datetime(2023, 1, 1, 12, 30, 0, tzinfo=_U),  # 8 A
    datetime(2023, 1, 1, 9, 30, 0, tzinfo=_U),  # 9 B (earliest B by time)
]

_FFILL_VALUES: list[float | None] = [
    1.0,  # 0 A
    None,  # 1 B leading null
    None,  # 2 A interior null after 1.0
    5.0,  # 3 B first non-null
    0.5,  # 4 A earliest by time
    None,  # 5 B trailing null
    None,  # 6 A trailing null
    None,  # 7 B interior null after 5.0
    9.0,  # 8 A
    None,  # 9 B earliest by time -> leading null
]


def _create_ffill_arrow_table() -> pa.Table:
    """Create the 10-row PyArrow table used by every ffill test."""
    return pa.table(
        {
            "id": pa.array(_FFILL_IDS, type=pa.int64()),
            "region": pa.array(_FFILL_REGIONS, type=pa.string()),
            "ts": pa.array(_FFILL_TIMESTAMPS, type=pa.timestamp("us", tz="UTC")),
            "value": pa.array(_FFILL_VALUES, type=pa.float64()),
        }
    )


# ---------------------------------------------------------------------------
# Expected values (offline-computed per the per-partition spec)
# ---------------------------------------------------------------------------

# Per-partition ffill (partition_by=["region"], order_by="ts"), in original row order.
# Differs from the naive whole-table fill at rows 1, 5, and 9 (partition bleed guard).
EXPECTED_FFILL: list[Any] = [
    1.0,  # 0 A passthrough
    None,  # 1 B leading null stays null
    1.0,  # 2 A interior null filled from 1.0 (NOT 5.0 -> no bleed from B)
    5.0,  # 3 B passthrough
    0.5,  # 4 A passthrough
    5.0,  # 5 B trailing null filled from 5.0
    9.0,  # 6 A trailing null filled from 9.0
    5.0,  # 7 B interior null filled from 5.0
    9.0,  # 8 A passthrough
    None,  # 9 B earliest by time -> leading null stays null
]

# Naive whole-table ffill (order_by="ts" only, no partition), in original row order.
# This is the EXPECTED output for the no-partition test, AND the wrong answer that
# the partition-aware result must NOT equal.
#
# Global ascending order by ts (now globally unique, deterministic for SQL):
#   id4(0.5) id9(None) id0(1.0) id1(None) id2(None) id3(5.0) id7(None) id8(9.0) id5(None) id6(None)
# Forward-filling that single sequence and mapping back to row order [0..9]:
NAIVE_WHOLE_TABLE_FFILL: list[Any] = [
    1.0,  # 0 (A, 10:00) own value
    1.0,  # 1 (B, 10:30) bleeds A's 1.0 across into B
    1.0,  # 2 (A, 11:00) carries A's 1.0 (same as partitioned)
    5.0,  # 3 (B, 11:30) own value
    0.5,  # 4 (A, 09:00) earliest overall, own value
    9.0,  # 5 (B, 13:00) bleeds A's 9.0 (12:30) across into B
    9.0,  # 6 (A, 13:30) latest overall, carries 9.0
    5.0,  # 7 (B, 12:00) carries B's 5.0 (same as partitioned)
    9.0,  # 8 (A, 12:30) own value
    0.5,  # 9 (B, 09:30) carries A's 0.5 (09:00) across into B
]


# ---------------------------------------------------------------------------
# Reusable test base class
# ---------------------------------------------------------------------------


class FfillTestBase(ReservedColumnsTestMixin, DataOpsTestBase):
    """Abstract base class for ffill-by-time framework tests.

    Subclasses combine this with a framework mixin (``PyArrowTestMixin``,
    ``PandasTestMixin``, etc.) and a one-liner ``implementation_class``
    classmethod returning the framework-specific feature group.

    ffill is a single-op operation (no op/unit matrix), so there is no
    ``supported_ops`` machinery here. All five backends support ffill
    natively; there are no rejections of supported inputs.
    """

    # -- ReservedColumnsTestMixin configuration --------------------------------

    @classmethod
    def reserved_columns_feature_name(cls) -> str:
        return "value__ffill"

    @classmethod
    def reserved_columns_order_by(cls) -> str | None:
        return "ts"

    @classmethod
    def reserved_columns_helper_names(cls) -> set[str]:
        return {"__mloda_rn__", "__mloda_rn"}

    @classmethod
    def reference_implementation_class(cls) -> Any:
        # Lazy import so this testing module imports cleanly even before the
        # production code exists (Red phase). The per-backend test files import
        # their implementation classes at module top level; that is where Red
        # fails with ImportError.
        from mloda.community.feature_groups.data_operations.row_preserving.ffill.pyarrow_ffill import (
            PyArrowFfill,
        )

        return PyArrowFfill

    # -- Setup: use the dedicated 10-row ffill fixture ----------------------

    def setup_method(self) -> None:
        """Override the canonical-fixture setup to use the dedicated 10-row table."""
        super().setup_method()  # connections + canonical data (mostly unused)
        self._arrow_table = _create_ffill_arrow_table()
        self.test_data = self.create_test_data(self._arrow_table)

    # -- Helpers ------------------------------------------------------------

    def _ffill_feature_set(self) -> FeatureSet:
        return make_feature_set("value__ffill", partition_by=["region"], order_by="ts")

    # -- Core per-partition value test --------------------------------------

    def test_basic_ffill_with_partition_and_order(self) -> None:
        """Per-partition ffill with partition_by=['region'] + order_by='ts'."""
        fs = self._ffill_feature_set()
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert isinstance(result, self.get_expected_type())
        col = self.extract_column(result, "value__ffill")
        self._assert_float_list_with_nulls(col, EXPECTED_FFILL)

    def test_leading_null_stays_null(self) -> None:
        """Leading nulls (before the first non-null in time order) stay NULL.

        Row 9 is B's earliest timestamp (09:30) and is null -> stays null.
        Row 1 (B, 10:30) follows only that leading null -> stays null too.
        """
        fs = self._ffill_feature_set()
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "value__ffill")
        assert col[9] is None, f"row 9 (B leading null) must stay None, got {col[9]!r}"
        assert col[1] is None, f"row 1 (B still leading) must stay None, got {col[1]!r}"

    def test_interior_and_trailing_null_filled(self) -> None:
        """Interior nulls after a non-null and trailing nulls are carried forward."""
        fs = self._ffill_feature_set()
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "value__ffill")
        # Interior: row 2 (A, 11:00) after A's 1.0 at 10:00 -> 1.0.
        assert float(col[2]) == pytest.approx(1.0), f"row 2 interior null -> 1.0, got {col[2]!r}"
        # Interior: row 7 (B, 12:00) after B's 5.0 at 11:30 -> 5.0.
        assert float(col[7]) == pytest.approx(5.0), f"row 7 interior null -> 5.0, got {col[7]!r}"
        # Trailing: row 6 (A, 13:30, last in A time order) -> 9.0.
        assert float(col[6]) == pytest.approx(9.0), f"row 6 trailing null -> 9.0, got {col[6]!r}"
        # Trailing: row 5 (B, 13:00, last in B time order) -> 5.0.
        assert float(col[5]) == pytest.approx(5.0), f"row 5 trailing null -> 5.0, got {col[5]!r}"

    def test_no_partition_whole_table_ffill(self) -> None:
        """With order_by only (no partition), ffill treats the whole table as one group."""
        fs = make_feature_set("value__ffill", order_by="ts")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "value__ffill")
        self._assert_float_list_with_nulls(col, NAIVE_WHOLE_TABLE_FFILL)

    def test_partition_aware_differs_from_naive(self) -> None:
        """Partition-aware fill must NOT equal the naive whole-table fill.

        This is the key test that catches partition bleed: a backend that
        ignores ``partition_by`` would produce ``NAIVE_WHOLE_TABLE_FFILL``.
        """
        fs = self._ffill_feature_set()
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "value__ffill")
        normalized = [None if v is None else float(v) for v in col]
        assert normalized == EXPECTED_FFILL, f"per-partition result mismatch: {normalized!r}"
        assert normalized != NAIVE_WHOLE_TABLE_FFILL, "ffill bled across partitions (partition_by ignored)"

    # -- Row-preserving semantics -------------------------------------------

    def test_output_rows_equal_input_rows(self) -> None:
        fs = self._ffill_feature_set()
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert self.get_row_count(result) == 10

    def test_original_row_order_preserved(self) -> None:
        """The passthrough ``id`` column must be unchanged in original row order."""
        fs = self._ffill_feature_set()
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        ids = self.extract_column(result, "id")
        assert [int(v) for v in ids] == _FFILL_IDS, f"row order changed: {ids!r}"

    def test_source_column_unchanged(self) -> None:
        """The source ``value`` column must pass through unchanged (only nulls get a NEW column)."""
        fs = self._ffill_feature_set()
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        src = self.extract_column(result, "value")
        self._assert_float_list_with_nulls(src, _FFILL_VALUES)

    def test_new_column_added(self) -> None:
        fs = self._ffill_feature_set()
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "value__ffill")
        assert len(col) == 10

    def test_result_has_correct_type(self) -> None:
        fs = self._ffill_feature_set()
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert isinstance(result, self.get_expected_type())

    # -- Option-based configuration -----------------------------------------

    def test_option_based_ffill(self) -> None:
        """Option-based configuration (no string pattern) produces the same result."""
        feature = Feature(
            "value__ffill",
            options=Options(
                context={
                    "in_features": "value",
                    "partition_by": ["region"],
                    "order_by": "ts",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "value__ffill")
        self._assert_float_list_with_nulls(col, EXPECTED_FFILL)

    # -- Cross-framework comparison -----------------------------------------

    def test_cross_framework_partitioned(self) -> None:
        """Compare partition-aware ffill against the PyArrow reference.

        Threads ``partition_by`` / ``order_by`` through BOTH the framework call
        and the reference call so the oracle also fills per partition.
        """
        self._compare_with_reference(
            "value__ffill",
            partition_by=["region"],
            order_by="ts",
        )

    def test_cross_framework_whole_table(self) -> None:
        """Compare whole-table ffill (no partition) against the PyArrow reference."""
        self._compare_with_reference(
            "value__ffill",
            partition_by=[],
            order_by="ts",
        )

    # -- Error / validation --------------------------------------------------

    def test_missing_source_column_raises_value_error(self) -> None:
        """A missing source value column must raise a clear ValueError naming the column.

        The table keeps ``id`` / ``region`` / ``ts`` so the error isolates the
        missing ``value`` column (not a missing order_by / partition column).
        """
        table = pa.table(
            {
                "id": pa.array(_FFILL_IDS, type=pa.int64()),
                "region": pa.array(_FFILL_REGIONS, type=pa.string()),
                "ts": pa.array(_FFILL_TIMESTAMPS, type=pa.timestamp("us", tz="UTC")),
            }
        )
        data = self.create_test_data(table)
        fs = make_feature_set("value__ffill", partition_by=["region"], order_by="ts")
        with pytest.raises(ValueError, match=r"(?i)value|missing|column"):
            self.implementation_class().calculate_feature(data, fs)

    def test_multi_column_in_features_rejected_at_calculate(self) -> None:
        """calculate_feature must reject features with multiple in_features (MAX_IN_FEATURES=1)."""
        feature = Feature(
            "bad_multi_col",
            options=Options(
                context={
                    "in_features": ["value", "other_value"],
                    "partition_by": ["region"],
                    "order_by": "ts",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        with pytest.raises(ValueError, match=r"(?i)at most 1|in_features|single"):
            self.implementation_class().calculate_feature(self.test_data, fs)
