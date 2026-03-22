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

from abc import ABC, abstractmethod
from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.user import Feature


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


class OffsetTestBase(ABC):
    """Abstract base class for offset framework tests."""

    ALL_OFFSET_TYPES = {"lag", "lead", "diff", "pct_change", "first_value", "last_value"}

    @classmethod
    def supported_offset_types(cls) -> set[str]:
        """Offset types this framework supports. Override to restrict."""
        return cls.ALL_OFFSET_TYPES

    @classmethod
    @abstractmethod
    def implementation_class(cls) -> Any:
        """Return the Offset implementation class to test."""

    @classmethod
    def pyarrow_implementation_class(cls) -> Any:
        """Return the PyArrow implementation class (reference)."""
        from mloda.community.feature_groups.data_operations.row_preserving.offset.pyarrow_offset import PyArrowOffset

        return PyArrowOffset

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
        """Return the expected type of the result."""

    def setup_method(self) -> None:
        self._arrow_table = PyArrowDataOpsTestDataCreator.create()
        self.test_data = self.create_test_data(self._arrow_table)

    def teardown_method(self) -> None:
        conn = getattr(self, "conn", None)
        if conn is not None:
            conn.close()

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

    def _compare_with_pyarrow(self, feature_name: str, partition_by: list[str], order_by: str) -> None:
        fs = make_feature_set(feature_name, partition_by, order_by)
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        ref = self.pyarrow_implementation_class().calculate_feature(self._arrow_table, fs)

        result_col = self.extract_column(result, feature_name)
        ref_col = extract_column(ref, feature_name)
        assert len(result_col) == len(ref_col)
        assert result_col == ref_col

    def test_cross_framework_lag(self) -> None:
        self._compare_with_pyarrow("value_int__lag_1_offset", ["region"], "value_int")

    def test_cross_framework_lead(self) -> None:
        self._compare_with_pyarrow("value_int__lead_1_offset", ["region"], "value_int")

    def test_cross_framework_first_value(self) -> None:
        self._compare_with_pyarrow("value_int__first_value_offset", ["region"], "value_int")

    def _skip_if_unsupported(self, offset_type: str) -> None:
        if offset_type not in self.supported_offset_types():
            pytest.skip(f"{offset_type} not supported by this framework")
