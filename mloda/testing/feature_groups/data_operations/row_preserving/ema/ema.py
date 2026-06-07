"""Shared test bases, dedicated fixture, and pinned expected values for EMA tests.

``ema`` computes an exponential moving average (exponential decay) of a value
column over time. Within each partition, rows are sorted by an ``order_by``
(time) column ascending, then an exponentially weighted mean is accumulated:

    ema[i] = alpha * x[i] + (1 - alpha) * ema[i-1]

with ``alpha = 2 / (span + 1)``, ``adjust=False``, and nulls SKIPPED in the
recurrence (a null input leaves the running ema unchanged and produces a NULL
output for that row). The first non-null seeds the recurrence. The operation is
ROW-PRESERVING: the result has the same rows in the same original order as the
input, with one new ``{col}__ema_{span}`` column appended.

Feature naming: ``{col}__ema_{span}`` where ``span`` is a positive integer
(e.g. ``value__ema_2``, ``value__ema_3``). The span is passed DIRECTLY to the
underlying library (pandas ``ewm(span=...)`` / polars ``ewm_mean(span=...)``);
backends must NOT pre-convert to alpha -- each library performs the identical
``span -> alpha`` mapping internally.

Null rules (pinned and empirically verified):

- A null input row produces a NULL output row.
- Leading nulls (before the first non-null in time order) produce NULL.
- An interior null is SKIPPED in the recurrence: the next non-null uses the
  prior ema, not a reset.
- Trailing nulls produce NULL.

NO LIVE REFERENCE ORACLE. Unlike ``ffill``, PyArrow cannot compute an
exponentially weighted mean, so it can NOT be the cross-framework reference.
Instead the expected values are PINNED literals computed offline from the
canonical pandas formula and BOTH compute backends (pandas, polars-lazy) are
asserted against the SAME literals. The pandas <-> polars span-form agreement
was verified empirically (elementwise, null-aware, within 1e-9) before pinning.

Backends:

- pandas + polars-lazy compute EMA natively (value tests via ``EmaTestBase``).
- pyarrow, duckdb, sqlite REJECT EMA with a clear ``ValueError`` (they have no
  native exponentially weighted compute and a recursive Python emulation is
  forbidden by the CFW-backend rule). They use ``EmaRejectionTestBase``, which
  inherits NONE of the value tests -- only a single rejection assertion.

Fixture row layout (12 rows, two interleaved partitions A / B in ROW order;
``id`` is the passthrough row-order witness). Per-partition TIME order differs
from ROW order so sorting actually matters, and all 12 timestamps are GLOBALLY
UNIQUE so SQL backends get a deterministic whole-table sort::

    id | region | ts hour | value | role (in partition TIME order)
    ---+--------+---------+-------+-----------------------------------------
    0  | A      | 02:00   | 2.0   | A time-rank 2 (recur: after 1.0)
    1  | B      | 07:00   | 10.0  | B time-rank 1 (first non-null / seed)
    2  | A      | 00:00   | None  | A time-rank 0 (leading null)
    3  | B      | 09:00   | None  | B time-rank 3 (interior null, skipped)
    4  | A      | 04:00   | 4.0   | A time-rank 4 (recur: after 2.0)
    5  | B      | 06:00   | None  | B time-rank 0 (leading null)
    6  | A      | 01:00   | 1.0   | A time-rank 1 (first non-null / seed)
    7  | B      | 11:00   | None  | B time-rank 5 (trailing null)
    8  | A      | 05:00   | None  | A time-rank 5 (trailing null)
    9  | B      | 08:00   | 20.0  | B time-rank 2 (recur: after 10.0)
    10 | A      | 03:00   | None  | A time-rank 3 (interior null, skipped)
    11 | B      | 10:00   | 40.0  | B time-rank 4 (recur: after 20.0)

    Per-partition TIME order (by ts ascending), values in that order:
      A: id2(None) id6(1.0) id0(2.0) id10(None) id4(4.0) id8(None)
      B: id5(None) id1(10.0) id9(20.0) id3(None) id11(40.0) id7(None)
    Each partition: leading null, seed, recur, interior-null-skipped, recur,
    trailing null -- so the recurrence accumulates visible depth in BOTH.
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
# Dedicated 12-row fixture (UTC timestamps, two interleaved partitions)
# ---------------------------------------------------------------------------

_EMA_IDS: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

_EMA_REGIONS: list[str] = ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]

# Hour-of-day per row (all globally unique -> deterministic SQL whole-table sort).
_EMA_TS_HOURS: list[int] = [2, 7, 0, 9, 4, 6, 1, 11, 5, 8, 3, 10]

_EMA_TIMESTAMPS: list[datetime] = [datetime(2023, 1, 1, h, 0, 0, tzinfo=_U) for h in _EMA_TS_HOURS]

_EMA_VALUES: list[float | None] = [
    2.0,  # 0  A
    10.0,  # 1  B seed
    None,  # 2  A leading null
    None,  # 3  B interior null
    4.0,  # 4  A
    None,  # 5  B leading null
    1.0,  # 6  A seed
    None,  # 7  B trailing null
    None,  # 8  A trailing null
    20.0,  # 9  B
    None,  # 10 A interior null
    40.0,  # 11 B
]


def _create_ema_arrow_table() -> pa.Table:
    """Create the 12-row PyArrow table used by every EMA test."""
    return pa.table(
        {
            "id": pa.array(_EMA_IDS, type=pa.int64()),
            "region": pa.array(_EMA_REGIONS, type=pa.string()),
            "ts": pa.array(_EMA_TIMESTAMPS, type=pa.timestamp("us", tz="UTC")),
            "value": pa.array(_EMA_VALUES, type=pa.float64()),
        }
    )


# ---------------------------------------------------------------------------
# Pinned expected values (offline-computed; pandas <-> polars span-form verified)
# ---------------------------------------------------------------------------
#
# Computed via the canonical pandas formula, per the per-partition spec, in
# ORIGINAL ROW ORDER:
#
#   for region, group in df.groupby("region"):
#       s = group.sort_values("ts")["value"]
#       ema = s.ewm(span=SPAN, adjust=False, ignore_na=True).mean().mask(s.isna())
#
# and the whole-table variant sorts the entire frame by ts (no groupby).
# Polars span-form ewm_mean(span=SPAN, adjust=False, ignore_nulls=True) agrees
# elementwise (null-aware, within 1e-9) with all four lists below.

# Per-partition EMA, span=2 (partition_by=["region"], order_by="ts").
EXPECTED_EMA_SPAN2: list[Any] = [
    1.6666666666666665,  # 0  A
    10.0,  # 1  B seed
    None,  # 2  A leading null
    None,  # 3  B interior null
    3.2222222222222223,  # 4  A
    None,  # 5  B leading null
    1.0,  # 6  A seed
    None,  # 7  B trailing null
    None,  # 8  A trailing null
    16.666666666666664,  # 9  B
    None,  # 10 A interior null
    32.22222222222222,  # 11 B
]

# Per-partition EMA, span=3.
EXPECTED_EMA_SPAN3: list[Any] = [
    1.5,  # 0  A
    10.0,  # 1  B seed
    None,  # 2  A leading null
    None,  # 3  B interior null
    2.75,  # 4  A
    None,  # 5  B leading null
    1.0,  # 6  A seed
    None,  # 7  B trailing null
    None,  # 8  A trailing null
    15.0,  # 9  B
    None,  # 10 A interior null
    27.5,  # 11 B
]

# Whole-table EMA, span=2 (order_by="ts" only, no partition). Differs from the
# per-partition result at rows 1, 9, 11 (partition bleed guard).
EXPECTED_EMA_WHOLE_SPAN2: list[Any] = [
    1.6666666666666665,  # 0
    7.7407407407407405,  # 1
    None,  # 2
    None,  # 3
    3.2222222222222223,  # 4
    None,  # 5
    1.0,  # 6
    None,  # 7
    None,  # 8
    15.913580246913579,  # 9
    None,  # 10
    31.971193415637856,  # 11
]

# Whole-table EMA, span=3.
EXPECTED_EMA_WHOLE_SPAN3: list[Any] = [
    1.5,  # 0
    6.375,  # 1
    None,  # 2
    None,  # 3
    2.75,  # 4
    None,  # 5
    1.0,  # 6
    None,  # 7
    None,  # 8
    13.1875,  # 9
    None,  # 10
    26.59375,  # 11
]


# ---------------------------------------------------------------------------
# Value / semantics test base (COMPUTE backends only: pandas, polars-lazy)
# ---------------------------------------------------------------------------


class EmaTestBase(ReservedColumnsTestMixin, DataOpsTestBase):
    """Reusable test base for EMA on backends that compute it NATIVELY.

    Subclasses combine this with a framework mixin (``PandasTestMixin``,
    ``PolarsLazyTestMixin``) and a one-liner ``implementation_class``
    classmethod returning the framework-specific feature group.

    There is NO live reference oracle (PyArrow cannot compute EMA); every value
    test asserts against PINNED literals, so this base is used ONLY by backends
    that actually support EMA. The reject backends use ``EmaRejectionTestBase``
    instead and inherit none of these value tests.
    """

    # -- ReservedColumnsTestMixin configuration --------------------------------

    @classmethod
    def reserved_columns_feature_name(cls) -> str:
        return "value__ema_2"

    @classmethod
    def reserved_columns_order_by(cls) -> str | None:
        return "ts"

    @classmethod
    def reserved_columns_helper_names(cls) -> set[str]:
        return {"__mloda_rn__"}

    # -- Setup: use the dedicated 12-row EMA fixture ------------------------

    def setup_method(self) -> None:
        """Override the canonical-fixture setup to use the dedicated 12-row table."""
        super().setup_method()  # connections + canonical data (mostly unused)
        self._arrow_table = _create_ema_arrow_table()
        self.test_data = self.create_test_data(self._arrow_table)

    # -- Helpers ------------------------------------------------------------

    def _ema_feature_set(self, span: int) -> FeatureSet:
        return make_feature_set(f"value__ema_{span}", partition_by=["region"], order_by="ts")

    # -- Core per-partition value tests -------------------------------------

    def test_ema_span2_per_partition(self) -> None:
        """Per-partition EMA span=2 matches pinned EXPECTED_EMA_SPAN2."""
        fs = self._ema_feature_set(2)
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert isinstance(result, self.get_expected_type())
        col = self.extract_column(result, "value__ema_2")
        self._assert_float_list_with_nulls(col, EXPECTED_EMA_SPAN2)

    def test_ema_span3_per_partition(self) -> None:
        """Per-partition EMA span=3 matches pinned EXPECTED_EMA_SPAN3."""
        fs = self._ema_feature_set(3)
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "value__ema_3")
        self._assert_float_list_with_nulls(col, EXPECTED_EMA_SPAN3)

    def test_ema_whole_table_span2(self) -> None:
        """With order_by only (no partition), EMA treats the whole table as one group."""
        fs = make_feature_set("value__ema_2", order_by="ts")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "value__ema_2")
        self._assert_float_list_with_nulls(col, EXPECTED_EMA_WHOLE_SPAN2)

    def test_ema_whole_table_span3(self) -> None:
        """Whole-table EMA span=3 matches pinned EXPECTED_EMA_WHOLE_SPAN3."""
        fs = make_feature_set("value__ema_3", order_by="ts")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "value__ema_3")
        self._assert_float_list_with_nulls(col, EXPECTED_EMA_WHOLE_SPAN3)

    def test_partition_aware_differs_from_whole_table(self) -> None:
        """Partition-aware EMA must NOT equal the whole-table EMA.

        Catches partition bleed: a backend that ignores ``partition_by`` would
        produce ``EXPECTED_EMA_WHOLE_SPAN2`` instead.
        """
        fs = self._ema_feature_set(2)
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "value__ema_2")
        normalized = [None if v is None else float(v) for v in col]
        # Per-partition pinned values match.
        for i, (a, e) in enumerate(zip(normalized, EXPECTED_EMA_SPAN2)):
            if e is None:
                assert a is None, f"row {i}: expected None, got {a!r}"
            else:
                assert a == pytest.approx(e), f"row {i}: {a!r} != {e!r}"
        # And the result is NOT the whole-table answer (differs at rows 1, 9, 11).
        assert normalized != EXPECTED_EMA_WHOLE_SPAN2, "EMA bled across partitions (partition_by ignored)"

    # -- Null semantics -----------------------------------------------------

    def test_leading_null_produces_null(self) -> None:
        """Leading nulls (before the first non-null in time order) stay NULL.

        Row 2 is A's earliest timestamp (00:00) and is null -> NULL.
        Row 5 is B's earliest timestamp (06:00) and is null -> NULL.
        """
        fs = self._ema_feature_set(2)
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "value__ema_2")
        assert col[2] is None, f"row 2 (A leading null) must be None, got {col[2]!r}"
        assert col[5] is None, f"row 5 (B leading null) must be None, got {col[5]!r}"

    def test_interior_null_skipped_in_recurrence(self) -> None:
        """An interior null is NULL in output AND skipped in the recurrence.

        For A (time order 1.0, 2.0, [null at 03:00], 4.0): the null row 10 is
        NULL, and the next non-null (row 4, value 4.0) uses the ema accumulated
        through 2.0 -- NOT a reset. EXPECTED_EMA_SPAN2[4] == 3.2222... encodes
        exactly that "next value uses prior ema across the skipped null".
        """
        fs = self._ema_feature_set(2)
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "value__ema_2")
        # Interior null rows are NULL.
        assert col[10] is None, f"row 10 (A interior null) must be None, got {col[10]!r}"
        assert col[3] is None, f"row 3 (B interior null) must be None, got {col[3]!r}"
        # The value AFTER the skipped null uses the prior ema, not a reset.
        assert float(col[4]) == pytest.approx(3.2222222222222223), f"row 4 should use prior ema, got {col[4]!r}"
        assert float(col[11]) == pytest.approx(32.22222222222222), f"row 11 should use prior ema, got {col[11]!r}"

    def test_trailing_null_produces_null(self) -> None:
        """Trailing nulls (after the last non-null in time order) stay NULL."""
        fs = self._ema_feature_set(2)
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "value__ema_2")
        assert col[8] is None, f"row 8 (A trailing null) must be None, got {col[8]!r}"
        assert col[7] is None, f"row 7 (B trailing null) must be None, got {col[7]!r}"

    def test_seed_is_first_non_null(self) -> None:
        """The first non-null in time order seeds the recurrence (ema == value)."""
        fs = self._ema_feature_set(2)
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "value__ema_2")
        # A seed: row 6 (01:00) value 1.0 -> ema 1.0.
        assert float(col[6]) == pytest.approx(1.0), f"A seed row 6 should be 1.0, got {col[6]!r}"
        # B seed: row 1 (07:00) value 10.0 -> ema 10.0.
        assert float(col[1]) == pytest.approx(10.0), f"B seed row 1 should be 10.0, got {col[1]!r}"

    # -- Row-preserving semantics -------------------------------------------

    def test_output_rows_equal_input_rows(self) -> None:
        fs = self._ema_feature_set(2)
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert self.get_row_count(result) == 12

    def test_original_row_order_preserved(self) -> None:
        """The passthrough ``id`` column must be unchanged in original row order."""
        fs = self._ema_feature_set(2)
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        ids = self.extract_column(result, "id")
        assert [int(v) for v in ids] == _EMA_IDS, f"row order changed: {ids!r}"

    def test_source_column_unchanged(self) -> None:
        """The source ``value`` column must pass through unchanged."""
        fs = self._ema_feature_set(2)
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        src = self.extract_column(result, "value")
        self._assert_float_list_with_nulls(src, _EMA_VALUES)

    def test_new_column_added(self) -> None:
        fs = self._ema_feature_set(2)
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        col = self.extract_column(result, "value__ema_2")
        assert len(col) == 12

    def test_result_has_correct_type(self) -> None:
        fs = self._ema_feature_set(2)
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert isinstance(result, self.get_expected_type())

    # -- Option-based configuration -----------------------------------------

    def test_option_based_ema(self) -> None:
        """Option-based configuration (no string pattern) produces the same result."""
        feature = Feature(
            "value__ema_2",
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
        col = self.extract_column(result, "value__ema_2")
        self._assert_float_list_with_nulls(col, EXPECTED_EMA_SPAN2)

    # -- Error / validation -------------------------------------------------

    def test_missing_source_column_raises_value_error(self) -> None:
        """A missing source value column must raise a clear ValueError.

        The table keeps ``id`` / ``region`` / ``ts`` so the error isolates the
        missing ``value`` column (not a missing order_by / partition column).
        """
        table = pa.table(
            {
                "id": pa.array(_EMA_IDS, type=pa.int64()),
                "region": pa.array(_EMA_REGIONS, type=pa.string()),
                "ts": pa.array(_EMA_TIMESTAMPS, type=pa.timestamp("us", tz="UTC")),
            }
        )
        data = self.create_test_data(table)
        fs = make_feature_set("value__ema_2", partition_by=["region"], order_by="ts")
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


# ---------------------------------------------------------------------------
# Rejection test base (REJECT backends: pyarrow, duckdb, sqlite)
# ---------------------------------------------------------------------------


class EmaRejectionTestBase(DataOpsTestBase):
    """Reusable base for backends that REJECT EMA up-front with a ValueError.

    PyArrow has no exponentially weighted compute; DuckDB and SQLite have no
    native EWM and a recursive / Python emulation is forbidden by the
    CFW-backend rule (see ``.claude/agents/green-agent.md`` "CFW Backend Rules"
    and the rejection precedents it cites). These backends must therefore reject
    EMA at validation time rather than computing it.

    This base inherits NONE of the value/semantics tests in ``EmaTestBase`` --
    only the single rejection assertion below. Subclasses combine it with a
    framework mixin (``PyArrowTestMixin`` / ``DuckdbTestMixin`` /
    ``SqliteTestMixin``) so ``create_test_data`` (and any connection lifecycle)
    comes from the mixin.
    """

    def setup_method(self) -> None:
        """Build the dedicated 12-row EMA fixture (mixins may create a conn first)."""
        super().setup_method()  # connections (duckdb/sqlite) + canonical data
        self._arrow_table = _create_ema_arrow_table()
        self.test_data = self.create_test_data(self._arrow_table)

    def test_ema_rejected_with_value_error(self) -> None:
        """calculate_feature must raise a clear ValueError naming the EMA limitation."""
        fs = make_feature_set("value__ema_2", partition_by=["region"], order_by="ts")
        with pytest.raises(ValueError, match=r"(?i)ema|exponential|not support|native"):
            self.implementation_class().calculate_feature(self.test_data, fs)
