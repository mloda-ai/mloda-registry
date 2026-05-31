"""Shared test base, dedicated fixture, and pinned expected values for sessionization.

``sessionize`` assigns an integer SESSION ID to each row by a gap-threshold
rule over a timestamp column. The source / in-feature column IS the timestamp
being sessionized. The operation is ROW-PRESERVING: the result has the same
rows in the same original order as the input, with one new
``{ts}__sessionize_{n}_{unit}`` integer column appended.

Feature naming: ``{ts}__sessionize_{n}_{unit}`` where ``n`` is a positive
integer and ``unit`` is one of ``minute`` / ``hour`` / ``day`` / ``week``
(e.g. ``ts__sessionize_30_minute``, ``ts__sessionize_1_hour``). The threshold
is ``n`` of ``unit`` expressed in seconds (minute=60, hour=3600, day=86400,
week=604800).

Semantics (pinned and verified):

- Sort by ``[*partition_by, order_by]``. ``order_by`` DEFAULTS to the named
  timestamp source column when absent. ``partition_by`` defaults to ``[]``
  (the whole table is a single stream).
- Within the sorted frame, a row STARTS A NEW SESSION if it is the first in
  its partition OR the gap to the previous row (in time order, within the
  partition) is STRICTLY GREATER than the threshold. Equal gap (gap ==
  threshold) stays in the SAME session (gap > threshold is the strict rule).
- ``session_id = cumsum(is_new) - 1`` over the sorted frame, producing a
  GLOBALLY-UNIQUE 0-based integer per row (session ids are not reset per
  partition; they are unique across the whole sorted frame).
- Output column dtype: integer (int64).

Nulls in the timestamp column are OUT OF SCOPE for these tests: a null gap is
ill-defined under a strict ``gap > threshold`` comparison (it is neither
clearly a new session nor a continuation), so the dedicated fixture below uses
globally-unique, non-null UTC timestamps. Backends decide null handling
separately; it is not pinned here.

NO PINNED-ONLY ORACLE. Unlike EMA, every backend computes sessionization
NATIVELY (like ffill), so PyArrow is the live cross-framework reference oracle
(``reference_implementation_class``). The expected session ids below are also
PINNED literals (computed offline from the algorithm above) so the spec is
fixed independent of any single backend.

Fixture row layout (9 rows, two interleaved users A / B in ROW order; ``id``
is the passthrough row-order witness). Per-user TIME order differs from ROW
order so sorting actually matters, and all 9 timestamps are GLOBALLY UNIQUE so
SQL backends get a deterministic whole-table sort::

    id | user | ts (H:M) | per-user time rank
    ---+------+----------+--------------------
    0  | A    | 11:30    | A: 10:00 < 10:20 < 10:50 < 11:30 < 11:35
    1  | B    | 09:01    | B: 09:01 < 10:02 < 10:27 < 12:03
    2  | A    | 10:00    |
    3  | B    | 12:03    |
    4  | A    | 10:50    |
    5  | B    | 10:27    |
    6  | A    | 11:35    |
    7  | B    | 10:02    |
    8  | A    | 10:20    |

    Per-user gaps (minutes), in TIME order:
      A: id2(10:00) -> id8(10:20)=20 -> id4(10:50)=30 -> id0(11:30)=40 -> id6(11:35)=5
      B: id1(09:01) -> id7(10:02)=61 -> id5(10:27)=25 -> id3(12:03)=96

    With a 30-minute threshold (strict gap > 30), per user:
      A: 10:00 new | 10:20 (+20<=30) same | 10:50 (+30<=30) same | 11:30 (+40>30) new | 11:35 (+5<=30) same
      B: 09:01 new | 10:02 (+61>30) new | 10:27 (+25<=30) same | 12:03 (+96>30) new
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
from mloda.user import Feature

_U = timezone.utc


# ---------------------------------------------------------------------------
# Dedicated 9-row fixture (UTC timestamps, two interleaved users)
# ---------------------------------------------------------------------------

_SESSION_IDS: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8]

_SESSION_USERS: list[str] = ["A", "B", "A", "B", "A", "B", "A", "B", "A"]

# (hour, minute) per row; all globally unique -> deterministic SQL whole-table sort.
_SESSION_HM: list[tuple[int, int]] = [
    (11, 30),  # 0 A
    (9, 1),  # 1 B
    (10, 0),  # 2 A
    (12, 3),  # 3 B
    (10, 50),  # 4 A
    (10, 27),  # 5 B
    (11, 35),  # 6 A
    (10, 2),  # 7 B
    (10, 20),  # 8 A
]

_SESSION_TIMESTAMPS: list[datetime] = [datetime(2023, 1, 1, h, m, 0, tzinfo=_U) for (h, m) in _SESSION_HM]


def _create_sessionization_arrow_table() -> pa.Table:
    """Create the 9-row PyArrow table used by every sessionization test."""
    return pa.table(
        {
            "id": pa.array(_SESSION_IDS, type=pa.int64()),
            "user": pa.array(_SESSION_USERS, type=pa.string()),
            "ts": pa.array(_SESSION_TIMESTAMPS, type=pa.timestamp("us", tz="UTC")),
        }
    )


# ---------------------------------------------------------------------------
# Pinned expected session ids (offline-computed per the algorithm above)
# ---------------------------------------------------------------------------
#
# session_id = cumsum(is_new) - 1 over the sorted frame, in ORIGINAL ROW order.
#
# Per-user, 30-minute threshold. New-session boundaries (sorted by user then ts):
#   B: 09:01(new s0) 10:02(+61>30 new s1) 10:27(+25 same s1) 12:03(+96>30 new s2)
#   A: 10:00(new s3) 10:20(+20 same s3) 10:50(+30 same s3) 11:30(+40>30 new s4) 11:35(+5 same s4)
# cumsum over [*partition_by="user", "ts"] sorts users alphabetically (B's "B"
# > "A"? no: "A" < "B"), so the global new-session order is A's stream then B's.
# Mapping ids back to row order [0..8]:

# Per-user 30-minute threshold (partition_by=["user"], ts__sessionize_30_minute).
EXPECTED_SESSION_30_MINUTE: list[int] = [1, 2, 0, 4, 0, 3, 1, 3, 0]

# Per-user 1-hour threshold (partition_by=["user"], ts__sessionize_1_hour).
EXPECTED_SESSION_1_HOUR: list[int] = [0, 1, 0, 3, 0, 2, 0, 2, 0]

# Whole-table 30-minute threshold (NO partition_by, order_by="ts").
EXPECTED_SESSION_30_MINUTE_WHOLE: list[int] = [2, 0, 1, 2, 1, 1, 2, 1, 1]


# ---------------------------------------------------------------------------
# Reusable test base class (all five backends compute natively)
# ---------------------------------------------------------------------------


class SessionizationTestBase(DataOpsTestBase):
    """Abstract base class for sessionization framework tests.

    Subclasses combine this with a framework mixin (``PandasTestMixin``,
    ``PyArrowTestMixin``, etc.) and a one-liner ``implementation_class``
    classmethod returning the framework-specific feature group.

    sessionization is a single-op operation (no op/unit support matrix), so
    there is no ``supported_ops`` machinery here. All five backends support
    sessionization natively; there are no rejections of supported inputs.
    """

    @classmethod
    def reference_implementation_class(cls) -> Any:
        # Lazy import so this testing module imports cleanly even before the
        # production code exists (Red phase). The per-backend test files import
        # their implementation classes at module top level; that is where Red
        # fails with ImportError.
        from mloda.community.feature_groups.data_operations.row_preserving.sessionization.pyarrow_sessionization import (
            PyArrowSessionization,
        )

        return PyArrowSessionization

    # -- Setup: use the dedicated 9-row sessionization fixture ---------------

    def setup_method(self) -> None:
        """Override the canonical-fixture setup to use the dedicated 9-row table."""
        super().setup_method()  # connections + canonical data (mostly unused)
        self._arrow_table = _create_sessionization_arrow_table()
        self.test_data = self.create_test_data(self._arrow_table)

    # -- Helpers ------------------------------------------------------------

    def _assert_int_list(self, actual: list[Any], expected: list[int]) -> None:
        assert len(actual) == len(expected), f"row count {len(actual)} != expected {len(expected)}"
        normalized = [None if v is None else int(v) for v in actual]
        assert normalized == expected, f"session ids {normalized!r} != expected {expected!r}"

    def _session_feature_set(
        self,
        n: int,
        unit: str,
        partition_by: list[str] | None = None,
        order_by: str | None = None,
    ) -> FeatureSet:
        """Build a sessionization FeatureSet.

        ``partition_by`` defaults to ``["user"]``; pass ``[]`` for the
        whole-table stream. ``order_by`` is left unset by default to exercise
        the "default order_by to the source timestamp column" behaviour.
        """
        if partition_by is None:
            partition_by = ["user"]
        return make_feature_set(
            f"ts__sessionize_{n}_{unit}",
            partition_by=partition_by,
            order_by=order_by,
        )

    # -- Core value tests ----------------------------------------------------

    def test_per_partition_30_minute(self) -> None:
        """Per-user 30-minute sessionization (order_by DEFAULTS to source ``ts``)."""
        fs = self._session_feature_set(30, "minute")  # no order_by -> default to source column
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert isinstance(result, self.get_expected_type())
        col = self.extract_column(result, "ts__sessionize_30_minute")
        self._assert_int_list(col, EXPECTED_SESSION_30_MINUTE)

    def test_per_partition_1_hour(self) -> None:
        """Per-user 1-hour sessionization (order_by passed EXPLICITLY)."""
        fs = self._session_feature_set(1, "hour", partition_by=["user"], order_by="ts")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "ts__sessionize_1_hour")
        self._assert_int_list(col, EXPECTED_SESSION_1_HOUR)

    def test_whole_table_30_minute(self) -> None:
        """With order_by only (no partition), sessionize treats the whole table as one stream."""
        fs = self._session_feature_set(30, "minute", partition_by=[], order_by="ts")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "ts__sessionize_30_minute")
        self._assert_int_list(col, EXPECTED_SESSION_30_MINUTE_WHOLE)

    def test_partition_aware_differs_from_whole_table(self) -> None:
        """Partition-aware result must match per-user pins AND NOT equal the whole-table list.

        Catches partition bleed: a backend that ignores ``partition_by`` would
        produce ``EXPECTED_SESSION_30_MINUTE_WHOLE`` instead.
        """
        fs = self._session_feature_set(30, "minute", partition_by=["user"], order_by="ts")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "ts__sessionize_30_minute")
        normalized = [int(v) for v in col]
        assert normalized == EXPECTED_SESSION_30_MINUTE, f"per-partition mismatch: {normalized!r}"
        assert normalized != EXPECTED_SESSION_30_MINUTE_WHOLE, "sessionization bled across partitions"

    # -- Row-preserving semantics -------------------------------------------

    def test_output_rows_equal_input_rows(self) -> None:
        fs = self._session_feature_set(30, "minute")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert self.get_row_count(result) == 9

    def test_original_row_order_preserved(self) -> None:
        """The passthrough ``id`` column must be unchanged in original row order."""
        fs = self._session_feature_set(30, "minute")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        ids = self.extract_column(result, "id")
        assert [int(v) for v in ids] == _SESSION_IDS, f"row order changed: {ids!r}"

    def test_source_ts_column_unchanged(self) -> None:
        """The source ``ts`` timestamp column must pass through unchanged."""
        fs = self._session_feature_set(30, "minute")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        src = self.extract_column(result, "ts")
        assert list(src) == list(_SESSION_TIMESTAMPS), f"ts column changed: {src!r}"

    def test_new_column_added(self) -> None:
        fs = self._session_feature_set(30, "minute")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "ts__sessionize_30_minute")
        assert len(col) == 9

    def test_result_has_correct_type(self) -> None:
        fs = self._session_feature_set(30, "minute")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert isinstance(result, self.get_expected_type())

    # -- Option-based configuration -----------------------------------------

    def test_option_based_sessionization(self) -> None:
        """Option-based configuration (no string-pattern order_by) produces the same result."""
        feature = Feature(
            "ts__sessionize_30_minute",
            options=Options(
                context={
                    "in_features": "ts",
                    "partition_by": ["user"],
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "ts__sessionize_30_minute")
        self._assert_int_list(col, EXPECTED_SESSION_30_MINUTE)

    # -- Cross-framework comparison -----------------------------------------

    def test_cross_framework_partitioned(self) -> None:
        """Compare partition-aware sessionization against the PyArrow reference."""
        self._compare_with_reference(
            "ts__sessionize_30_minute",
            partition_by=["user"],
            order_by="ts",
        )

    # -- Error / validation --------------------------------------------------

    def test_missing_source_column_raises_value_error(self) -> None:
        """A missing source ``ts`` column must raise a clear ValueError.

        The table keeps ``id`` / ``user`` so the error isolates the missing
        ``ts`` column.
        """
        table = pa.table(
            {
                "id": pa.array(_SESSION_IDS, type=pa.int64()),
                "user": pa.array(_SESSION_USERS, type=pa.string()),
            }
        )
        data = self.create_test_data(table)
        fs = make_feature_set("ts__sessionize_30_minute", partition_by=["user"], order_by="ts")
        with pytest.raises(ValueError, match=r"(?i)ts|missing|column"):
            self.implementation_class().calculate_feature(data, fs)

    def test_multi_column_in_features_rejected_at_calculate(self) -> None:
        """calculate_feature must reject features with multiple in_features (MAX_IN_FEATURES=1)."""
        feature = Feature(
            "bad_multi_col",
            options=Options(
                context={
                    "in_features": ["ts", "other_ts"],
                    "partition_by": ["user"],
                    "order_by": "ts",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        with pytest.raises(ValueError, match=r"(?i)at most 1|in_features|single"):
            self.implementation_class().calculate_feature(self.test_data, fs)
