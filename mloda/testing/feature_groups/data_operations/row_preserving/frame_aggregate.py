"""Shared test base class, data, and helpers for frame aggregate tests.

Expected values are computed from the canonical 12-row dataset in
DataOperationsTestDataCreator, partitioned by the 'region' column
(A=rows 0-3, B=rows 4-7, C=rows 8-10, None=row 11) and ordered by
'value_int'.

The ``FrameAggregateTestBase`` class provides concrete test methods that
any framework implementation inherits by subclassing and implementing a
small set of abstract methods. This follows the same pattern as
``WindowAggregationTestBase``.
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

# Rolling sum (window 3) on value_int, partitioned by region, ordered by value_int.
# Group A sorted by value_int: (-5, 0, 10, 20)
#   row1(-5): window=[-5] -> -5
#   row2(0):  window=[-5, 0] -> -5
#   row0(10): window=[-5, 0, 10] -> 5
#   row3(20): window=[0, 10, 20] -> 30
# Group B sorted by value_int: (30, 50, 60, None)
#   row6(30): window=[30] -> 30
#   row5(50): window=[30, 50] -> 80
#   row7(60): window=[30, 50, 60] -> 140
#   row4(None): window=[50, 60, None] -> 110 (null skipped)
# Group C sorted by value_int: (15, 15, 40)
#   row8(15): window=[15] -> 15
#   row9(15): window=[15, 15] -> 30
#   row10(40): window=[15, 15, 40] -> 70
# Group None: (-10)
#   row11(-10): window=[-10] -> -10
#
# Output ordered by original row index: [row0, row1, row2, ..., row11]
EXPECTED_ROLLING_SUM_3: list[int] = [5, -5, -5, 30, 110, 80, 30, 140, 15, 30, 70, -10]

# Cumulative sum on value_int, partitioned by region, ordered by value_int.
# Group A: (-5) -> (-5+0) -> (-5+0+10) -> (-5+0+10+20)
#   = -5, -5, 5, 25
# Group B: (30) -> (30+50) -> (30+50+60) -> (30+50+60+None)
#   = 30, 80, 140, 140
# Group C: (15) -> (15+15) -> (15+15+40) = 15, 30, 70
# Group None: -10
#
# Output ordered by original row index:
EXPECTED_CUMSUM: list[int] = [5, -5, -5, 25, 140, 80, 30, 140, 15, 30, 70, -10]

# Rolling avg (window 2) on value_int, partitioned by region, ordered by value_int.
# Group A sorted: (-5, 0, 10, 20)
#   row1: [-5] -> -5.0
#   row2: [-5, 0] -> -2.5
#   row0: [0, 10] -> 5.0
#   row3: [10, 20] -> 15.0
# Group B sorted: (30, 50, 60, None)
#   row6: [30] -> 30.0
#   row5: [30, 50] -> 40.0
#   row7: [50, 60] -> 55.0
#   row4: [60, None] -> 60.0
# Group C sorted: (15, 15, 40)
#   row8: [15] -> 15.0
#   row9: [15, 15] -> 15.0
#   row10: [15, 40] -> 27.5
# Group None: -10.0
EXPECTED_ROLLING_AVG_2: list[float] = [5.0, -5.0, -2.5, 15.0, 60.0, 40.0, 30.0, 55.0, 15.0, 15.0, 27.5, -10.0]

# Expanding avg on value_int, partitioned by region, ordered by value_int.
# Group A sorted: (-5, 0, 10, 20)
#   row1: [-5] -> -5.0
#   row2: [-5, 0] -> -2.5
#   row0: [-5, 0, 10] -> 1.6667
#   row3: [-5, 0, 10, 20] -> 6.25
# Group B sorted: (30, 50, 60, None)
#   row6: [30] -> 30.0
#   row5: [30, 50] -> 40.0
#   row7: [30, 50, 60] -> 46.6667
#   row4: [30, 50, 60, None] -> 46.6667
# Group C sorted: (15, 15, 40)
#   row8: [15] -> 15.0
#   row9: [15, 15] -> 15.0
#   row10: [15, 15, 40] -> 23.3333
# Group None: -10.0
EXPECTED_EXPANDING_AVG: list[float] = [
    5.0 / 3.0,
    -5.0,
    -2.5,
    6.25,
    140.0 / 3.0,
    40.0,
    30.0,
    140.0 / 3.0,
    15.0,
    15.0,
    70.0 / 3.0,
    -10.0,
]


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def _is_null(value: Any) -> bool:
    """Check if a value is null (None or NaN)."""
    if value is None:
        return True
    import math

    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _assert_values_with_nulls(actual: list[Any], expected: list[Any]) -> None:
    """Assert two lists are equal, treating None and NaN as equivalent nulls."""
    assert len(actual) == len(expected), f"length {len(actual)} != {len(expected)}"
    for i, (a, e) in enumerate(zip(actual, expected)):
        if _is_null(e):
            assert _is_null(a), f"row {i}: expected null, got {a}"
        else:
            assert a == pytest.approx(e, rel=1e-6), f"row {i}: {a} != {e}"


def extract_column(result: Any, column_name: str) -> list[Any]:
    """Extract a column from a result object as a Python list."""
    if isinstance(result, pa.Table):
        return list(result.column(column_name).to_pylist())
    if hasattr(result, "to_arrow_table"):
        arrow_table = result.to_arrow_table()
        return list(arrow_table.column(column_name).to_pylist())
    if hasattr(result, "collect"):
        df = result.collect()
        return list(df[column_name].to_list())
    return list(result[column_name])


def make_feature_set(
    feature_name: str,
    partition_by: list[str],
    order_by: str,
) -> FeatureSet:
    """Build a FeatureSet with partition_by and order_by options."""
    context: dict[str, Any] = {"partition_by": partition_by, "order_by": order_by}
    feature = Feature(
        feature_name,
        options=Options(context=context),
    )
    fs = FeatureSet()
    fs.add(feature)
    return fs


# ---------------------------------------------------------------------------
# Reusable test base class
# ---------------------------------------------------------------------------


class FrameAggregateTestBase(ABC):
    """Abstract base class for frame aggregate framework tests.

    Subclasses implement abstract methods to wire up their framework,
    then inherit concrete test methods covering rolling, cumulative,
    expanding, row preservation, time window rejection (where applicable),
    and cross-framework comparison against PyArrow.
    """

    # -- Abstract methods subclasses must implement --------------------------

    @classmethod
    @abstractmethod
    def implementation_class(cls) -> Any:
        """Return the FrameAggregate implementation class to test."""

    @classmethod
    def pyarrow_implementation_class(cls) -> Any:
        """Return the PyArrow implementation class (reference for cross-framework comparison)."""
        from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.pyarrow_frame_aggregate import (
            PyArrowFrameAggregate,
        )

        return PyArrowFrameAggregate

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

    @classmethod
    def supports_time_frame(cls) -> bool:
        """Whether this framework supports frame_type='time'. Default: False."""
        return False

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

    # -- Rolling tests -------------------------------------------------------

    def test_rolling_sum_window_3(self) -> None:
        """Rolling sum with window size 3, partitioned by region, ordered by value_int."""
        fs = make_feature_set("value_int__sum_rolling_3", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__sum_rolling_3")
        assert result_col == EXPECTED_ROLLING_SUM_3

    def test_rolling_avg_window_2(self) -> None:
        """Rolling avg with window size 2, partitioned by region, ordered by value_int."""
        fs = make_feature_set("value_int__avg_rolling_2", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__avg_rolling_2")
        assert result_col == pytest.approx(EXPECTED_ROLLING_AVG_2, rel=1e-3)

    # -- Cumulative tests ----------------------------------------------------

    def test_cumulative_sum(self) -> None:
        """Cumulative sum, partitioned by region, ordered by value_int."""
        fs = make_feature_set("value_int__cumsum", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__cumsum")
        assert result_col == EXPECTED_CUMSUM

    # -- Expanding tests -----------------------------------------------------

    def test_expanding_avg(self) -> None:
        """Expanding avg, partitioned by region, ordered by value_int."""
        fs = make_feature_set("value_int__expanding_avg", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__expanding_avg")
        assert result_col == pytest.approx(EXPECTED_EXPANDING_AVG, rel=1e-3)

    # -- Row preservation ----------------------------------------------------

    def test_output_rows_equal_input_rows(self) -> None:
        """Output must have exactly 12 rows, same as input."""
        fs = make_feature_set("value_int__sum_rolling_3", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert self.get_row_count(result) == 12

    def test_result_has_correct_type(self) -> None:
        """The result of calculate_feature must be the expected framework type."""
        fs = make_feature_set("value_int__cumsum", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())

    # -- Time window support tests -------------------------------------------

    def test_time_frame_match_rejected_when_unsupported(self) -> None:
        """Frameworks that do not support time frames must reject them at match time."""
        if self.supports_time_frame():
            pytest.skip("This framework supports time frames")

        options = Options(context={"partition_by": ["region"], "order_by": "timestamp"})
        assert not self.implementation_class().match_feature_group_criteria("value_int__avg_7_day_window", options)

    def test_time_frame_config_rejected_when_unsupported(self) -> None:
        """Config-based time frame features must be rejected at match time when unsupported."""
        if self.supports_time_frame():
            pytest.skip("This framework supports time frames")

        options = Options(
            context={
                "aggregation_type": "sum",
                "frame_type": "time",
                "frame_size": 2,
                "frame_unit": "day",
                "partition_by": ["region"],
                "order_by": "timestamp",
            }
        )
        assert not self.implementation_class().match_feature_group_criteria("my_time_sum", options)

    # -- Cross-framework comparison ------------------------------------------

    def _compare_with_pyarrow(
        self, feature_name: str, partition_by: list[str], order_by: str, use_approx: bool = False
    ) -> None:
        """Run the feature through this framework and PyArrow, assert results match."""
        fs = make_feature_set(feature_name, partition_by, order_by)
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        ref = self.pyarrow_implementation_class().calculate_feature(self._arrow_table, fs)

        result_col = self.extract_column(result, feature_name)
        ref_col = extract_column(ref, feature_name)

        assert len(result_col) == len(ref_col), f"row count {len(result_col)} != reference {len(ref_col)}"
        if use_approx:
            for i, (ref_val, fw_val) in enumerate(zip(ref_col, result_col)):
                if ref_val is None:
                    assert fw_val is None, f"row {i}: expected None, got {fw_val}"
                else:
                    assert fw_val == pytest.approx(ref_val, rel=1e-6), f"row {i}: {fw_val} != reference {ref_val}"
        else:
            assert result_col == ref_col

    def test_cross_framework_rolling_sum(self) -> None:
        """Rolling sum must match PyArrow reference."""
        self._compare_with_pyarrow("value_int__sum_rolling_3", ["region"], "value_int")

    def test_cross_framework_cumsum(self) -> None:
        """Cumulative sum must match PyArrow reference."""
        self._compare_with_pyarrow("value_int__cumsum", ["region"], "value_int")

    def test_cross_framework_expanding_avg(self) -> None:
        """Expanding avg must match PyArrow reference."""
        self._compare_with_pyarrow("value_int__expanding_avg", ["region"], "value_int", use_approx=True)

    def test_cross_framework_rolling_avg(self) -> None:
        """Rolling avg must match PyArrow reference."""
        self._compare_with_pyarrow("value_int__avg_rolling_2", ["region"], "value_int", use_approx=True)

    # -- Edge case tests -----------------------------------------------------

    def test_rolling_window_size_1(self) -> None:
        """Rolling with window_size=1 should return the value itself."""
        fs = make_feature_set("value_int__sum_rolling_1", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__sum_rolling_1")
        # Window of 1 means each value is its own window, so the sum equals the original value.
        # Original row order: [10, -5, 0, 20, None, 50, 30, 60, 15, 15, 40, -10]
        # Nulls in sum_rolling_1 should produce None (or NaN in Pandas).
        expected = [10, -5, 0, 20, None, 50, 30, 60, 15, 15, 40, -10]
        _assert_values_with_nulls(result_col, expected)

    def test_all_null_values_returns_null(self) -> None:
        """When all values in the source column are null, results should be null."""
        all_null_table = pa.table(
            {
                "region": ["A", "A", "A"],
                "value_int": [None, None, None],
            }
        )
        data = self.create_test_data(all_null_table)
        fs = make_feature_set("value_int__cumsum", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(data, fs)

        assert self.get_row_count(result) == 3
        result_col = self.extract_column(result, "value_int__cumsum")
        for i, val in enumerate(result_col):
            assert _is_null(val), f"row {i}: expected null, got {val}"

    def test_window_larger_than_partition(self) -> None:
        """Rolling window larger than partition should include all rows in the partition."""
        fs = make_feature_set("value_int__sum_rolling_100", ["region"], "value_int")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__sum_rolling_100")
        # With window 100, each row sees all preceding rows in its partition.
        # This is equivalent to cumulative sum.
        assert result_col == EXPECTED_CUMSUM

    def test_multiple_null_order_by_values(self) -> None:
        """Two or more null order_by values must not crash during sorting."""
        table = pa.table(
            {
                "region": ["A", "A", "A", "A"],
                "ts": [None, 1, None, 2],
                "value": [100, 10, 200, 20],
            }
        )
        data = self.create_test_data(table)
        fs = make_feature_set("value__cumsum", ["region"], "ts")
        result = self.implementation_class().calculate_feature(data, fs)

        result_col = self.extract_column(result, "value__cumsum")

        # Sorted order: ts=1 (10), ts=2 (20), ts=None (100), ts=None (200)
        # Cumsum:        10,       30,          130,            330
        # Map back to original positions: row1->10, row3->30, nulls get 130 and 330
        assert result_col[1] == 10
        assert result_col[3] == 30
        assert set([result_col[0], result_col[2]]) == {130, 330}
