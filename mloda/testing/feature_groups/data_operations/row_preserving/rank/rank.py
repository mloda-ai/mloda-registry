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
5 abstract methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.user import Feature


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


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def extract_column(result: Any, column_name: str) -> list[Any]:
    """Extract a column from a result object as a Python list."""
    if isinstance(result, pa.Table):
        return list(result.column(column_name).to_pylist())
    arrow_table = result.to_arrow_table()
    return list(arrow_table.column(column_name).to_pylist())


def make_feature_set(feature_name: str, partition_by: list[str], order_by: str) -> FeatureSet:
    """Build a FeatureSet with partition_by and order_by options."""
    feature = Feature(
        feature_name,
        options=Options(context={"partition_by": partition_by, "order_by": order_by}),
    )
    fs = FeatureSet()
    fs.add(feature)
    return fs


# ---------------------------------------------------------------------------
# Reusable test base class
# ---------------------------------------------------------------------------


class RankTestBase(ABC):
    """Abstract base class for rank framework tests.

    Subclasses implement 5 abstract methods to wire up their framework,
    then inherit concrete test methods for free.
    """

    ALL_RANK_TYPES = {"row_number", "rank", "dense_rank", "percent_rank"}

    @classmethod
    def supported_rank_types(cls) -> set[str]:
        """Rank types this framework supports. Override to restrict."""
        return cls.ALL_RANK_TYPES

    # -- Abstract methods subclasses must implement --------------------------

    @classmethod
    @abstractmethod
    def implementation_class(cls) -> Any:
        """Return the Rank implementation class to test."""

    @classmethod
    def pyarrow_implementation_class(cls) -> Any:
        """Return the PyArrow implementation class (reference for cross-framework comparison)."""
        from mloda.community.feature_groups.data_operations.row_preserving.rank.pyarrow_rank import (
            PyArrowRank,
        )

        return PyArrowRank

    @abstractmethod
    def create_test_data(self, arrow_table: pa.Table) -> Any:
        """Convert the standard PyArrow test table to the framework's native format."""

    @abstractmethod
    def extract_column(self, result: Any, column_name: str) -> list[Any]:
        """Extract a column from the result as a Python list."""

    @abstractmethod
    def get_row_count(self, result: Any) -> int:
        """Return the number of rows in the result."""

    @abstractmethod
    def get_expected_type(self) -> Any:
        """Return the expected type of the result (for isinstance checks)."""

    # -- Setup / teardown ----------------------------------------------------

    def setup_method(self) -> None:
        """Create test data from the canonical 12-row dataset."""
        self._arrow_table = PyArrowDataOpsTestDataCreator.create()
        self.test_data = self.create_test_data(self._arrow_table)

    def teardown_method(self) -> None:
        """Close self.conn if it was set by a connection-based subclass."""
        conn = getattr(self, "conn", None)
        if conn is not None:
            conn.close()

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

    # -- Cross-framework comparison (matches PyArrow reference) --------------

    def _compare_with_pyarrow(self, feature_name: str, partition_by: list[str], order_by: str) -> None:
        """Run the feature through this framework and PyArrow, assert results match."""
        fs = make_feature_set(feature_name, partition_by, order_by)
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        ref = self.pyarrow_implementation_class().calculate_feature(self._arrow_table, fs)

        result_col = self.extract_column(result, feature_name)
        ref_col = extract_column(ref, feature_name)

        assert len(result_col) == len(ref_col), f"row count {len(result_col)} != reference {len(ref_col)}"
        assert result_col == ref_col

    def test_cross_framework_row_number(self) -> None:
        """Row number must match PyArrow reference."""
        self._compare_with_pyarrow("value_int__row_number_ranked", ["region"], "value_int")

    def test_cross_framework_rank(self) -> None:
        """Rank must match PyArrow reference."""
        self._compare_with_pyarrow("value_int__rank_ranked", ["region"], "value_int")

    def test_cross_framework_dense_rank(self) -> None:
        """Dense rank must match PyArrow reference."""
        self._compare_with_pyarrow("value_int__dense_rank_ranked", ["region"], "value_int")

    def test_cross_framework_percent_rank(self) -> None:
        """Percent rank must match PyArrow reference."""
        self._skip_if_unsupported("percent_rank")
        self._compare_with_pyarrow_approx("value_int__percent_rank_ranked", ["region"], "value_int")

    def test_cross_framework_ntile(self) -> None:
        """Ntile must match PyArrow reference."""
        self._compare_with_pyarrow("value_int__ntile_2_ranked", ["region"], "value_int")

    # -- Helper methods ------------------------------------------------------

    def _compare_with_pyarrow_approx(self, feature_name: str, partition_by: list[str], order_by: str) -> None:
        """Like _compare_with_pyarrow but uses pytest.approx for float comparison."""
        fs = make_feature_set(feature_name, partition_by, order_by)
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        ref = self.pyarrow_implementation_class().calculate_feature(self._arrow_table, fs)

        result_col = self.extract_column(result, feature_name)
        ref_col = extract_column(ref, feature_name)

        assert len(result_col) == len(ref_col), f"row count {len(result_col)} != reference {len(ref_col)}"
        for i, (actual, expected) in enumerate(zip(result_col, ref_col)):
            assert actual == pytest.approx(expected, rel=1e-6), f"row {i}: {actual} != {expected}"

    def _skip_if_unsupported(self, rank_type: str) -> None:
        if rank_type not in self.supported_rank_types():
            pytest.skip(f"{rank_type} not supported by this framework")
