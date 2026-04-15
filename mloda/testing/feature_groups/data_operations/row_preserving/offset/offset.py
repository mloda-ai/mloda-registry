"""Shared test base class for offset tests.

Expected values are computed from the canonical 12-row dataset,
partitioned by 'region' and ordered by 'value_int'.

Group compositions (ordered by value_int, nulls last):
  A: [-5, 0, 10, 20]  (rows 1, 2, 0, 3)
  B: [30, 50, 60, None] (rows 6, 5, 7, 4)
  C: [15, 15, 40]      (rows 8, 9, 10)
  None: [-10]           (row 11)
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.collision import CollisionTestMixin

__all__ = [
    "EXPECTED_FIRST_VALUE",
    "EXPECTED_LAG_1",
    "EXPECTED_LAST_VALUE",
    "EXPECTED_LEAD_1",
    "OffsetTestBase",
    "extract_column",
    "make_feature_set",
]


# ---------------------------------------------------------------------------
# Expected values
# ---------------------------------------------------------------------------
# Lag 1 (previous value within partition, ordered by value_int):
#   A: [None, -5, 0, 10]  mapped to rows [0,1,2,3] => [0, None, -5, 10]
#   row 0 (val 10, pos 2): lag=0,  row 1 (val -5, pos 0): lag=None
#   row 2 (val 0, pos 1): lag=-5,  row 3 (val 20, pos 3): lag=10
EXPECTED_LAG_1 = [0, None, -5, 10, 60, 30, None, 50, None, 15, 15, None]

# Lead 1 (next value within partition):
#   A: [0, 10, 20, None]  mapped: row 0=20, row 1=0, row 2=10, row 3=None
EXPECTED_LEAD_1 = [20, 0, 10, None, None, 60, 50, None, 15, 40, None, None]

# First value (first non-null in partition, ordered by value_int):
EXPECTED_FIRST_VALUE = [-5, -5, -5, -5, 30, 30, 30, 30, 15, 15, 15, -10]

# Last value (last non-null in partition, ordered by value_int):
#   B: values=[30,50,60,None] => last non-null=60
EXPECTED_LAST_VALUE = [20, 20, 20, 20, 60, 60, 60, 60, 40, 40, 40, -10]


class OffsetTestBase(CollisionTestMixin, DataOpsTestBase):
    """Abstract base class for offset framework tests."""

    ALL_OFFSET_TYPES = {"lag", "lead", "diff", "pct_change", "first_value", "last_value"}

    @classmethod
    def supported_offset_types(cls) -> set[str]:
        """Offset types this framework supports. Override to restrict."""
        return cls.ALL_OFFSET_TYPES

    # -- CollisionTestMixin configuration -------------------------------------

    @classmethod
    def collision_feature_name(cls) -> str:
        return "value_int__lag_1_offset"

    @classmethod
    def collision_partition_by(cls) -> list[str] | None:
        return ["region"]

    @classmethod
    def collision_order_by(cls) -> str | None:
        return "value_int"

    @classmethod
    def reference_implementation_class(cls) -> Any:
        """Return the reference implementation class."""
        from mloda.testing.feature_groups.data_operations.row_preserving.offset.reference import ReferenceOffset

        return ReferenceOffset

    # -- Concrete tests ------------------------------------------------------

    def test_lag_1(self) -> None:
        """Lag 1 of value_int partitioned by region, ordered by value_int."""
        fs = make_feature_set("value_int__lag_1_offset", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12
        result_col = self.extract_column(result, "value_int__lag_1_offset")
        assert result_col == EXPECTED_LAG_1

    def test_lead_1(self) -> None:
        """Lead 1 of value_int partitioned by region, ordered by value_int."""
        fs = make_feature_set("value_int__lead_1_offset", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12
        result_col = self.extract_column(result, "value_int__lead_1_offset")
        assert result_col == EXPECTED_LEAD_1

    def test_first_value(self) -> None:
        """First value of value_int partitioned by region, ordered by value_int."""
        fs = make_feature_set("value_int__first_value_offset", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12
        result_col = self.extract_column(result, "value_int__first_value_offset")
        assert result_col == EXPECTED_FIRST_VALUE

    def test_last_value(self) -> None:
        """Last value of value_int partitioned by region, ordered by value_int."""
        fs = make_feature_set("value_int__last_value_offset", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12
        result_col = self.extract_column(result, "value_int__last_value_offset")
        assert result_col == EXPECTED_LAST_VALUE

    def test_diff_1(self) -> None:
        """Diff 1 of value_int partitioned by region, ordered by value_int."""
        self._skip_if_unsupported("diff")
        fs = make_feature_set("value_int__diff_1_offset", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12
        result_col = self.extract_column(result, "value_int__diff_1_offset")
        # A: [-5, 0, 10, 20] => diffs: [None, 5, 10, 10]
        # row 0 (val 10, pos 2): diff = 10-0 = 10
        # row 1 (val -5, pos 0): diff = None (no prev)
        # row 2 (val 0, pos 1): diff = 0-(-5) = 5
        # row 3 (val 20, pos 3): diff = 20-10 = 10
        assert result_col[0] == 10  # 10 - 0
        assert result_col[1] is None
        assert result_col[2] == 5  # 0 - (-5)
        assert result_col[3] == 10  # 20 - 10

    def test_pct_change_1(self) -> None:
        """Pct change 1 of value_int partitioned by region, ordered by value_int."""
        self._skip_if_unsupported("pct_change")
        fs = make_feature_set("value_int__pct_change_1_offset", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12
        result_col = self.extract_column(result, "value_int__pct_change_1_offset")
        # A: [-5, 0, 10, 20] => pct_change: [None, -1.0, None(zero denom), 1.0]
        # row 0 (val 10, pos 2): prev=0 => None (zero denominator)
        # row 1 (val -5, pos 0): None (no prev)
        # row 2 (val 0, pos 1): (0 - (-5)) / (-5) = -1.0
        # row 3 (val 20, pos 3): (20 - 10) / 10 = 1.0
        assert result_col[0] is None  # zero denominator
        assert result_col[1] is None  # first in partition
        assert result_col[2] == pytest.approx(-1.0)
        assert result_col[3] == pytest.approx(1.0)
        # B: [30, 50, 60, None]
        # row 4 (None, pos 3): None (null current)
        # row 5 (50, pos 1): (50-30)/30 = 2/3
        # row 6 (30, pos 0): None (first in partition)
        # row 7 (60, pos 2): (60-50)/50 = 0.2
        assert result_col[4] is None
        assert result_col[5] == pytest.approx(2.0 / 3.0)
        assert result_col[6] is None
        assert result_col[7] == pytest.approx(0.2)
        # C: [15, 15, 40]
        # row 8 (15, pos 0): None (first)
        # row 9 (15, pos 1): (15-15)/15 = 0.0
        # row 10 (40, pos 2): (40-15)/15 = 5/3
        assert result_col[8] is None
        assert result_col[9] == pytest.approx(0.0)
        assert result_col[10] == pytest.approx(25.0 / 15.0)
        # None group: single row
        assert result_col[11] is None

    def test_null_policy_edge_null(self) -> None:
        """NullPolicy.EDGE_NULL: lag at start of partition produces null."""
        fs = make_feature_set("value_int__lag_1_offset", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__lag_1_offset")
        # Row 1 (val -5, first in A) should have lag=None
        assert result_col[1] is None
        # Row 6 (val 30, first non-null in B) should have lag=None
        assert result_col[6] is None
        # Row 11 (only row in None group) should have lag=None
        assert result_col[11] is None

    def test_output_rows_equal_input_rows(self) -> None:
        """Output must have exactly 12 rows."""
        fs = make_feature_set("value_int__lag_1_offset", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert self.get_row_count(result) == 12

    def test_new_column_added(self) -> None:
        """The offset result column should be added to the output."""
        fs = make_feature_set("value_int__lag_1_offset", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_cols = self.extract_column(result, "value_int__lag_1_offset")
        assert len(result_cols) == 12

    def test_result_has_correct_type(self) -> None:
        """The result must be the expected framework type."""
        fs = make_feature_set("value_int__lag_1_offset", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert isinstance(result, self.get_expected_type())

    # -- Cross-framework comparison ------------------------------------------

    def test_cross_framework_lag(self) -> None:
        self._compare_with_reference("value_int__lag_1_offset", partition_by=["region"], order_by="value_int")

    def test_cross_framework_lead(self) -> None:
        self._compare_with_reference("value_int__lead_1_offset", partition_by=["region"], order_by="value_int")

    def test_cross_framework_first_value(self) -> None:
        self._compare_with_reference("value_int__first_value_offset", partition_by=["region"], order_by="value_int")

    def test_cross_framework_last_value(self) -> None:
        self._compare_with_reference("value_int__last_value_offset", partition_by=["region"], order_by="value_int")

    def test_cross_framework_diff(self) -> None:
        self._skip_if_unsupported("diff")
        self._compare_with_reference("value_int__diff_1_offset", partition_by=["region"], order_by="value_int")

    def test_cross_framework_pct_change(self) -> None:
        self._skip_if_unsupported("pct_change")
        self._compare_with_reference(
            "value_int__pct_change_1_offset", partition_by=["region"], order_by="value_int", use_approx=True
        )

    # -- All-null column tests -----------------------------------------------

    def test_all_null_column_lag(self) -> None:
        """Lag on an all-null column should produce all None."""
        table = pa.table(
            {
                "region": ["A", "A", "A"],
                "score": pa.array([None, None, None], type=pa.int64()),
            }
        )
        data = self.create_test_data(table)
        fs = make_feature_set("score__lag_1_offset", ["region"], "score")
        result = self.implementation_class().calculate_feature(data, fs)

        result_col = self.extract_column(result, "score__lag_1_offset")
        assert all(v is None for v in result_col), f"expected all None, got {result_col}"

    def test_all_null_column_first_value(self) -> None:
        """First value on an all-null column should produce all None."""
        table = pa.table(
            {
                "region": ["A", "A", "A"],
                "score": pa.array([None, None, None], type=pa.int64()),
            }
        )
        data = self.create_test_data(table)
        fs = make_feature_set("score__first_value_offset", ["region"], "score")
        result = self.implementation_class().calculate_feature(data, fs)

        result_col = self.extract_column(result, "score__first_value_offset")
        assert all(v is None for v in result_col), f"expected all None, got {result_col}"

    # -- Option-based config tests -------------------------------------------

    def test_option_based_lag(self) -> None:
        """Option-based configuration (not string pattern) produces the same result as pattern."""
        from mloda.core.abstract_plugins.components.feature_set import FeatureSet
        from mloda.core.abstract_plugins.components.options import Options
        from mloda.user import Feature

        feature = Feature(
            "my_lag_result",
            options=Options(
                context={
                    "offset_type": "lag_1",
                    "in_features": "value_int",
                    "partition_by": ["region"],
                    "order_by": "value_int",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "my_lag_result")
        assert result_col == EXPECTED_LAG_1

    def test_unsupported_offset_type_raises(self) -> None:
        """Calling calculate_feature with an unknown offset type should raise ValueError."""
        from mloda.core.abstract_plugins.components.feature_set import FeatureSet
        from mloda.core.abstract_plugins.components.options import Options
        from mloda.user import Feature

        feature = Feature(
            "value_int__evil_type_offset",
            options=Options(
                context={
                    "partition_by": ["region"],
                    "order_by": "value_int",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        with pytest.raises((ValueError, KeyError)):
            self.implementation_class().calculate_feature(self.test_data, fs)

    # -- Tier 3: Partition / order_by null tests ------------------------------

    def test_null_partition_key(self) -> None:
        """Null in partition_by forms its own group. Row 11 has region=None."""
        fs = make_feature_set("value_int__first_value_offset", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__first_value_offset")
        # Row 11 has region=None, value_int=-10. Only member of its group.
        # first_value of a single-row group = that value.
        assert result_col[11] == -10

    def test_multi_key_partition_lag(self) -> None:
        """Lag partitioned by [region, category] produces correct grouping."""
        fs = make_feature_set("value_int__lag_1_offset", ["region", "category"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__lag_1_offset")
        # Group A/X: values [10, 0] sorted by value_int -> [0, 10]
        #   row 2 (val 0, pos 0): lag=None
        #   row 0 (val 10, pos 1): lag=0
        assert result_col[2] is None
        assert result_col[0] == 0

    def test_null_order_by_key(self) -> None:
        """Null in order_by column should rank last (nulls last)."""
        table = pa.table(
            {
                "region": ["A", "A", "A"],
                "ts": [None, 1, 2],
                "value": [100, 10, 20],
            }
        )
        data = self.create_test_data(table)
        fs = make_feature_set("value__lag_1_offset", ["region"], "ts")
        result = self.implementation_class().calculate_feature(data, fs)

        result_col = self.extract_column(result, "value__lag_1_offset")
        # Sorted by ts: 1->10, 2->20, None->100
        # Lag: [None, 10, 20]
        # Map back: row 0 (ts=None) -> lag=20, row 1 (ts=1) -> lag=None, row 2 (ts=2) -> lag=10
        assert result_col[1] is None  # first in sorted order has no predecessor

    # -- Row-order preservation ------------------------------------------------

    def test_row_order_preserved(self) -> None:
        """Original columns must remain in input row order after offset.

        PyArrow parity: without the ROW_NUMBER + restore pattern, SQL
        backends return rows sorted by the window ORDER BY clause.

        Compares (value_int, category) tuples to detect tied-row swaps
        (rows 8 and 9 both have value_int=15).
        """
        fs = make_feature_set("value_int__lag_1_offset", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        input_id = list(
            zip(
                self.extract_column(self.test_data, "value_int"),
                self.extract_column(self.test_data, "category"),
            )
        )
        output_id = list(
            zip(
                self.extract_column(result, "value_int"),
                self.extract_column(result, "category"),
            )
        )
        assert output_id == input_id
