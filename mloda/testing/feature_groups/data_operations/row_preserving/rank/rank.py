"""Shared test base class, data, and helpers for rank tests.

Expected values are computed from the canonical 12-row dataset in
DataOperationsTestDataCreator, partitioned by the 'region' column and
ordered by 'value_int'.

Group compositions (ordered by value_int, nulls last):
  A: [-5, 0, 10, 20]  (rows 1, 2, 0, 3)
  B: [30, 50, 60, None] (rows 6, 5, 7, 4)  -- null last
  C: [15, 15, 40]      (rows 8, 9, 10)     -- tie at 15
  None: [-10]           (row 11)

The ``RankTestBase`` class provides concrete test methods
that any framework implementation inherits by subclassing and implementing
the abstract adapter methods from ``DataOpsTestBase``.
"""

from __future__ import annotations

from typing import Any

import pytest
import pyarrow as pa

from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set


# ---------------------------------------------------------------------------
# Expected values (module-level constants)
# ---------------------------------------------------------------------------
# Row order in the 12-row dataset (by index):
#   region:    [A,  A,  A,  A,  B,    B,  B,  B,  C,  C,  C,  None]
#   value_int: [10, -5, 0,  20, None, 50, 30, 60, 15, 15, 40, -10]
#
# Within each partition, ordered by value_int ASC (nulls last):
#   A: -5(1), 0(2), 10(3), 20(4)   => row indices 1, 2, 0, 3
#   B: 30(6), 50(5), 60(7), None(4) => row indices 6, 5, 7, 4
#   C: 15(8), 15(9), 40(10)         => row indices 8, 9, 10
#   None: -10(11)                    => row index 11

# row_number: unique sequential (1-based), ties broken by insertion order
EXPECTED_ROW_NUMBER = [3, 1, 2, 4, 4, 2, 1, 3, 1, 2, 3, 1]

# rank: same rank for ties, gaps after
EXPECTED_RANK = [3, 1, 2, 4, 4, 2, 1, 3, 1, 1, 3, 1]

# dense_rank: same rank for ties, no gaps
EXPECTED_DENSE_RANK = [3, 1, 2, 4, 4, 2, 1, 3, 1, 1, 2, 1]

# top_3: ROW_NUMBER() OVER (PARTITION BY region ORDER BY value_int DESC NULLS LAST) <= 3
# Returns boolean: True if row is in the top 3 by value_int within its partition.
#   A (DESC): 20, 10, 0, -5 => top 3 = {20, 10, 0}
#   B (DESC): 60, 50, 30, None => top 3 = {60, 50, 30}
#   C (DESC): 40, 15, 15 => all 3 qualify
#   None: -10 => qualifies (N >= group_size)
EXPECTED_TOP_3 = [True, False, True, True, False, True, True, True, True, True, True, True]

# bottom_2: ROW_NUMBER() OVER (PARTITION BY region ORDER BY value_int ASC NULLS LAST) <= 2
# Returns boolean: True if row is in the bottom 2 by value_int within its partition.
#   A (ASC): -5, 0, 10, 20 => bottom 2 = {-5, 0}
#   B (ASC): 30, 50, 60, None => bottom 2 = {30, 50}
#   C (ASC): 15, 15, 40 => bottom 2 = {15, 15}
#   None: -10 => qualifies (N >= group_size)
EXPECTED_BOTTOM_2 = [False, True, True, False, False, True, True, False, True, True, False, True]


# ---------------------------------------------------------------------------
# Reusable test base class
# ---------------------------------------------------------------------------


class RankTestBase(DataOpsTestBase):
    """Abstract base class for rank framework tests.

    Subclasses implement the abstract adapter methods from ``DataOpsTestBase``
    to wire up their framework, then inherit concrete test methods for free.
    """

    ALL_RANK_TYPES = {"row_number", "rank", "dense_rank", "percent_rank"}

    @classmethod
    def supported_rank_types(cls) -> set[str]:
        """Rank types this framework supports. Override to restrict."""
        return cls.ALL_RANK_TYPES

    # -- PyArrow reference override --------------------------------------------

    @classmethod
    def pyarrow_implementation_class(cls) -> Any:
        """Return the PyArrow implementation class (reference for cross-framework comparison)."""
        from mloda.community.feature_groups.data_operations.row_preserving.rank.pyarrow_rank import (
            PyArrowRank,
        )

        return PyArrowRank

    # -- Concrete test methods (inherited for free) --------------------------

    def test_row_number_ranked(self) -> None:
        """Row number of value_int partitioned by region, ordered by value_int."""
        fs = make_feature_set("value_int__row_number_ranked", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__row_number_ranked")
        assert result_col == EXPECTED_ROW_NUMBER

    def test_rank_ranked(self) -> None:
        """Standard rank of value_int partitioned by region, ordered by value_int."""
        fs = make_feature_set("value_int__rank_ranked", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__rank_ranked")
        assert result_col == EXPECTED_RANK

    def test_dense_rank_ranked(self) -> None:
        """Dense rank of value_int partitioned by region, ordered by value_int."""
        fs = make_feature_set("value_int__dense_rank_ranked", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__dense_rank_ranked")
        assert result_col == EXPECTED_DENSE_RANK

    def test_percent_rank_ranked(self) -> None:
        """Percent rank of value_int partitioned by region, ordered by value_int."""
        self._skip_if_unsupported("percent_rank")
        fs = make_feature_set("value_int__percent_rank_ranked", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__percent_rank_ranked")
        # A: 4 rows => (rank-1)/(4-1): [-5=>0/3=0.0, 0=>1/3, 10=>2/3, 20=>3/3=1.0]
        assert result_col[0] == pytest.approx(2 / 3, rel=1e-6)  # row 0: value 10, rank 3
        assert result_col[1] == pytest.approx(0.0, rel=1e-6)  # row 1: value -5, rank 1
        assert result_col[2] == pytest.approx(1 / 3, rel=1e-6)  # row 2: value 0, rank 2
        assert result_col[3] == pytest.approx(1.0, rel=1e-6)  # row 3: value 20, rank 4
        # None group: single row => percent_rank = 0.0
        assert result_col[11] == pytest.approx(0.0, rel=1e-6)

    def test_ntile_ranked(self) -> None:
        """Ntile_2 of value_int partitioned by region, ordered by value_int."""
        fs = make_feature_set("value_int__ntile_2_ranked", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__ntile_2_ranked")
        # A: 4 rows, ntile(2) => [1,1,2,2] ordered by value
        # row 0 (value 10, rank 3) => bucket 2
        # row 1 (value -5, rank 1) => bucket 1
        # row 2 (value 0, rank 2) => bucket 1
        # row 3 (value 20, rank 4) => bucket 2
        assert result_col[0] == 2
        assert result_col[1] == 1
        assert result_col[2] == 1
        assert result_col[3] == 2
        # None group: 1 row, ntile(2) => 1
        assert result_col[11] == 1

    def test_top_n_ranked(self) -> None:
        """Top 3 of value_int partitioned by region, ordered by value_int."""
        fs = make_feature_set("value_int__top_3_ranked", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__top_3_ranked")
        assert result_col == EXPECTED_TOP_3

    def test_bottom_n_ranked(self) -> None:
        """Bottom 2 of value_int partitioned by region, ordered by value_int."""
        fs = make_feature_set("value_int__bottom_2_ranked", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__bottom_2_ranked")
        assert result_col == EXPECTED_BOTTOM_2

    def test_null_policy_nulls_last(self) -> None:
        """NullPolicy.NULLS_LAST: null values in order_by column rank last."""
        fs = make_feature_set("value_int__row_number_ranked", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__row_number_ranked")
        # Row 4 has value_int=None in group B. It should rank last (4).
        assert result_col[4] == 4

    def test_output_rows_equal_input_rows(self) -> None:
        """Output must have exactly 12 rows, same as input."""
        fs = make_feature_set("value_int__row_number_ranked", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert self.get_row_count(result) == 12

    def test_new_column_added(self) -> None:
        """The rank result column should be added to the output."""
        fs = make_feature_set("value_int__rank_ranked", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_cols = self.extract_column(result, "value_int__rank_ranked")
        assert len(result_cols) == 12

    def test_result_has_correct_type(self) -> None:
        """The result of calculate_feature must be the expected framework type."""
        fs = make_feature_set("value_int__row_number_ranked", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())

    # -- Edge case tests ------------------------------------------------------

    def test_ntile_1_all_bucket_1(self) -> None:
        """Ntile_1: every row gets bucket 1."""
        fs = make_feature_set("value_int__ntile_1_ranked", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__ntile_1_ranked")
        assert all(v == 1 for v in result_col)

    def test_ntile_n_exceeds_group_size(self) -> None:
        """Ntile_10 on groups smaller than 10: all buckets in 1..10, all unique per group."""
        fs = make_feature_set("value_int__ntile_10_ranked", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        region_col = self.extract_column(result, "region")
        result_col = self.extract_column(result, "value_int__ntile_10_ranked")

        # Group by region and check bucket values
        groups: dict[Any, list[int]] = {}
        for region, bucket in zip(region_col, result_col):
            groups.setdefault(region, []).append(bucket)

        for region, buckets in groups.items():
            # All buckets must be in range 1..10
            assert all(1 <= b <= 10 for b in buckets), f"region={region}: buckets out of range: {buckets}"
            # Each row in the group gets a distinct bucket
            assert len(set(buckets)) == len(buckets), f"region={region}: expected unique buckets, got {buckets}"

    def test_rank_all_ties(self) -> None:
        """Group C has two tied values (15, 15). Rank and dense_rank must agree on those rows."""
        fs = make_feature_set("value_int__rank_ranked", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__rank_ranked")
        # Rows 8 and 9 are both value_int=15 in group C: both get rank 1
        assert result_col[8] == 1
        assert result_col[9] == 1

    def test_top_n_exceeds_group_size(self) -> None:
        """Top 10 on groups smaller than 10: all rows are True."""
        fs = make_feature_set("value_int__top_10_ranked", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__top_10_ranked")
        assert all(v is True for v in result_col)

    def test_bottom_n_exceeds_group_size(self) -> None:
        """Bottom 10 on groups smaller than 10: all rows are True."""
        fs = make_feature_set("value_int__bottom_10_ranked", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__bottom_10_ranked")
        assert all(v is True for v in result_col)

    def test_top_1_selects_maximum(self) -> None:
        """Top 1: only the row with the highest value in each partition is True."""
        fs = make_feature_set("value_int__top_1_ranked", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__top_1_ranked")
        # A: max=20 at row 3 -> True, rest False
        assert result_col[3] is True
        assert result_col[0] is False  # val=10
        assert result_col[1] is False  # val=-5
        # B: max=60 at row 7 -> True
        assert result_col[7] is True
        assert result_col[6] is False  # val=30
        # None group: single row -> True
        assert result_col[11] is True

    # -- Cross-framework comparison (matches PyArrow reference) --------------

    def test_cross_framework_row_number(self) -> None:
        """Row number must match PyArrow reference."""
        self._compare_with_pyarrow("value_int__row_number_ranked", partition_by=["region"], order_by="value_int")

    def test_cross_framework_rank(self) -> None:
        """Rank must match PyArrow reference."""
        self._compare_with_pyarrow("value_int__rank_ranked", partition_by=["region"], order_by="value_int")

    def test_cross_framework_dense_rank(self) -> None:
        """Dense rank must match PyArrow reference."""
        self._compare_with_pyarrow("value_int__dense_rank_ranked", partition_by=["region"], order_by="value_int")

    def test_cross_framework_percent_rank(self) -> None:
        """Percent rank must match PyArrow reference."""
        self._skip_if_unsupported("percent_rank")
        self._compare_with_pyarrow(
            "value_int__percent_rank_ranked", partition_by=["region"], order_by="value_int", use_approx=True
        )

    def test_cross_framework_ntile(self) -> None:
        """Ntile must match PyArrow reference."""
        self._compare_with_pyarrow("value_int__ntile_2_ranked", partition_by=["region"], order_by="value_int")

    def test_cross_framework_top_n(self) -> None:
        """Top N must match PyArrow reference."""
        self._compare_with_pyarrow("value_int__top_3_ranked", partition_by=["region"], order_by="value_int")

    def test_cross_framework_bottom_n(self) -> None:
        """Bottom N must match PyArrow reference."""
        self._compare_with_pyarrow("value_int__bottom_2_ranked", partition_by=["region"], order_by="value_int")

    # -- All-null column tests -----------------------------------------------

    def test_all_null_column_row_number(self) -> None:
        """Row number on an all-null order_by column should still produce valid ranks."""
        table = pa.table(
            {
                "region": ["A", "A", "A"],
                "score": pa.array([None, None, None], type=pa.int64()),
            }
        )
        data = self.create_test_data(table)
        fs = make_feature_set("score__row_number_ranked", ["region"], "score")
        result = self.implementation_class().calculate_feature(data, fs)

        result_col = self.extract_column(result, "score__row_number_ranked")
        assert len(result_col) == 3
        # All values are null/tied, so row_number should assign 1, 2, 3
        assert sorted(result_col) == [1, 2, 3]

    def test_all_null_column_rank(self) -> None:
        """Rank on an all-null order_by column should assign rank 1 to all (all tied)."""
        table = pa.table(
            {
                "region": ["A", "A", "A"],
                "score": pa.array([None, None, None], type=pa.int64()),
            }
        )
        data = self.create_test_data(table)
        fs = make_feature_set("score__rank_ranked", ["region"], "score")
        result = self.implementation_class().calculate_feature(data, fs)

        result_col = self.extract_column(result, "score__rank_ranked")
        assert len(result_col) == 3
        # All nulls are tied, so all get rank 1
        assert all(v == 1 for v in result_col)

    # -- Option-based config tests -------------------------------------------

    def test_option_based_row_number(self) -> None:
        """Option-based configuration (not string pattern) produces the same result."""
        from mloda.core.abstract_plugins.components.feature_set import FeatureSet
        from mloda.core.abstract_plugins.components.options import Options
        from mloda.user import Feature

        feature = Feature(
            "my_row_number",
            options=Options(
                context={
                    "rank_type": "row_number",
                    "in_features": "value_int",
                    "partition_by": ["region"],
                    "order_by": "value_int",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "my_row_number")
        assert result_col == EXPECTED_ROW_NUMBER

    def test_option_based_top_n(self) -> None:
        """Option-based top_N produces the same result as string pattern."""
        from mloda.core.abstract_plugins.components.feature_set import FeatureSet
        from mloda.core.abstract_plugins.components.options import Options
        from mloda.user import Feature

        feature = Feature(
            "my_top_3",
            options=Options(
                context={
                    "rank_type": "top_3",
                    "in_features": "value_int",
                    "partition_by": ["region"],
                    "order_by": "value_int",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "my_top_3")
        assert result_col == EXPECTED_TOP_3

    def test_unsupported_rank_type_raises(self) -> None:
        """Calling calculate_feature with an unknown rank type should raise."""
        from mloda.core.abstract_plugins.components.feature_set import FeatureSet
        from mloda.core.abstract_plugins.components.options import Options
        from mloda.user import Feature

        feature = Feature(
            "value_int__evil_type_ranked",
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

    def test_null_partition_key_rank(self) -> None:
        """Null in partition_by forms its own group. Row 11 has region=None."""
        fs = make_feature_set("value_int__row_number_ranked", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__row_number_ranked")
        # Row 11 has region=None, single row in its group -> row_number = 1
        assert result_col[11] == 1

    def test_multi_key_partition_rank(self) -> None:
        """Rank partitioned by [region, category] produces correct grouping."""
        fs = make_feature_set("value_int__row_number_ranked", ["region", "category"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__row_number_ranked")
        # Group A/X: values [10, 0] sorted -> [0, 10], row_number = [1, 2]
        #   row 2 (val 0): row_number = 1
        #   row 0 (val 10): row_number = 2
        assert result_col[2] == 1
        assert result_col[0] == 2

    def test_null_order_by_produces_valid_rank(self) -> None:
        """Null in order_by column should rank last (nulls last)."""
        table = pa.table(
            {
                "region": ["A", "A", "A"],
                "ts": [None, 1, 2],
                "value": [100, 10, 20],
            }
        )
        data = self.create_test_data(table)
        fs = make_feature_set("value__row_number_ranked", ["region"], "ts")
        result = self.implementation_class().calculate_feature(data, fs)

        result_col = self.extract_column(result, "value__row_number_ranked")
        # Sorted by ts: 1->row_num 1, 2->row_num 2, None->row_num 3
        # Map back: row 0 (ts=None) -> 3, row 1 (ts=1) -> 1, row 2 (ts=2) -> 2
        assert result_col[1] == 1  # ts=1, first in sorted order
        assert result_col[2] == 2  # ts=2, second
        assert result_col[0] == 3  # ts=None, last (nulls last)

    def test_top_n_string_order_by(self) -> None:
        """Top 2 with a string order_by column: DESC sort must not use negation."""
        table = pa.table(
            {
                "region": ["A", "A", "A", "A"],
                "label": ["cherry", "apple", "banana", None],
            }
        )
        data = self.create_test_data(table)
        fs = make_feature_set("label__top_2_ranked", ["region"], "label")
        result = self.implementation_class().calculate_feature(data, fs)

        result_col = self.extract_column(result, "label__top_2_ranked")
        # DESC with nulls last: cherry(0), banana(2), apple(1), None(3)
        # Top 2 = rows 0 and 2 -> True; rows 1 and 3 -> False
        assert result_col == [True, False, True, False]

    def test_bottom_n_string_order_by(self) -> None:
        """Bottom 2 with a string order_by column: ASC sort on non-numeric types."""
        table = pa.table(
            {
                "region": ["A", "A", "A", "A"],
                "label": ["cherry", "apple", "banana", None],
            }
        )
        data = self.create_test_data(table)
        fs = make_feature_set("label__bottom_2_ranked", ["region"], "label")
        result = self.implementation_class().calculate_feature(data, fs)

        result_col = self.extract_column(result, "label__bottom_2_ranked")
        # ASC with nulls last: apple(1), banana(2), cherry(0), None(3)
        # Bottom 2 = rows 1 and 2 -> True; rows 0 and 3 -> False
        assert result_col == [False, True, True, False]

    # -- Helper methods ------------------------------------------------------

    def _skip_if_unsupported(self, rank_type: str) -> None:
        if rank_type not in self.supported_rank_types():
            pytest.skip(f"{rank_type} not supported by this framework")
