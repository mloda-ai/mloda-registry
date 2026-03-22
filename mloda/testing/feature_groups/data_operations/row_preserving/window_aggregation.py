"""Shared test base class, data, and helpers for window aggregation tests.

Expected values are computed from the canonical 12-row dataset in
DataOperationsTestDataCreator, partitioned by the 'region' column
(A=rows 0-3, B=rows 4-7, C=rows 8-10, None=row 11).

The ``WindowAggregationTestBase`` class provides 16 concrete test methods
(10 per-framework + 5 cross-framework comparison + 1 column check) that
any framework implementation inherits by subclassing and implementing
5 abstract methods. This follows the same pattern as mloda core's
``DataFrameTestBase`` in ``tests/test_plugins/compute_framework/test_tooling/``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pyarrow as pa
import pytest

from mloda.core.abstract_plugins.components.feature_set import FeatureSet
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.data_creator import PyArrowDataOpsTestDataCreator
from mloda.user import Feature


# ---------------------------------------------------------------------------
# Expected values (module-level constants, also usable standalone)
# ---------------------------------------------------------------------------

EXPECTED_SUM_BY_REGION: list[int] = [25, 25, 25, 25, 140, 140, 140, 140, 70, 70, 70, -10]
EXPECTED_AVG_BY_REGION: list[float] = [
    6.25,
    6.25,
    6.25,
    6.25,
    46.667,
    46.667,
    46.667,
    46.667,
    23.333,
    23.333,
    23.333,
    -10.0,
]
EXPECTED_COUNT_BY_REGION: list[int] = [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 1]
EXPECTED_MIN_BY_REGION: list[int] = [-5, -5, -5, -5, 30, 30, 30, 30, 15, 15, 15, -10]
EXPECTED_MAX_BY_REGION: list[int] = [20, 20, 20, 20, 60, 60, 60, 60, 40, 40, 40, -10]

# EC-016: Group B avg when one member (row 4) is null.
GROUP_B_AVG_EXPECTED: float = 140.0 / 3.0

# EC-019: Row 11 has region=None, value_int=-10. Forms its own group.
NULL_GROUP_SUM_EXPECTED: int = -10


# ---------------------------------------------------------------------------
# Standalone helpers (for cross-framework tests and other uses)
# ---------------------------------------------------------------------------


def extract_column(result: Any, column_name: str) -> list[Any]:
    """Extract a column from a result object as a Python list.

    Handles pa.Table (direct .column() access) and relation types
    (DuckdbRelation, SqliteRelation) that expose .to_arrow_table().
    """
    if isinstance(result, pa.Table):
        return list(result.column(column_name).to_pylist())
    arrow_table = result.to_arrow_table()
    return list(arrow_table.column(column_name).to_pylist())


def make_feature_set(feature_name: str, partition_by: list[str]) -> FeatureSet:
    """Build a FeatureSet with partition_by options."""
    feature = Feature(
        feature_name,
        options=Options(context={"partition_by": partition_by}),
    )
    fs = FeatureSet()
    fs.add(feature)
    return fs


# ---------------------------------------------------------------------------
# Reusable test base class
# ---------------------------------------------------------------------------


class WindowAggregationTestBase(ABC):
    """Abstract base class for window aggregation framework tests.

    Subclasses implement 5 abstract methods to wire up their framework,
    then inherit 16 concrete test methods for free (10 per-framework +
    5 cross-framework comparison against PyArrow + 1 column check).

    Simple frameworks (PyArrow, Pandas, Polars) need ~15 lines.
    Connection-based frameworks (DuckDB, SQLite) need ~25 lines
    (add setup_method/teardown_method for connection lifecycle).
    """

    # -- Abstract methods subclasses must implement --------------------------

    @classmethod
    @abstractmethod
    def implementation_class(cls) -> Any:
        """Return the WindowAggregation implementation class to test."""

    @classmethod
    def pyarrow_implementation_class(cls) -> Any:
        """Return the PyArrow implementation class (reference for cross-framework comparison)."""
        from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.pyarrow_window_aggregation import (
            PyArrowWindowAggregation,
        )

        return PyArrowWindowAggregation

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
        """Create test data from the canonical 12-row dataset.

        Connection-based subclasses should create their connection as
        ``self.conn`` first, then call ``super().setup_method()``.
        """
        self._arrow_table = PyArrowDataOpsTestDataCreator.create()
        self.test_data = self.create_test_data(self._arrow_table)

    def teardown_method(self) -> None:
        """Close self.conn if it was set by a connection-based subclass."""
        conn = getattr(self, "conn", None)
        if conn is not None:
            conn.close()

    # -- Concrete test methods (inherited for free) --------------------------

    def test_sum_groupby_region(self) -> None:
        """Sum of value_int partitioned by region, broadcast back to every row."""
        fs = make_feature_set("value_int__sum_groupby", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__sum_groupby")
        assert result_col == EXPECTED_SUM_BY_REGION

    def test_avg_groupby_region(self) -> None:
        """Average of value_int partitioned by region, broadcast back to every row."""
        fs = make_feature_set("value_int__avg_groupby", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__avg_groupby")
        assert result_col == pytest.approx(EXPECTED_AVG_BY_REGION, rel=1e-3)

    def test_count_groupby_region(self) -> None:
        """Count of non-null value_int partitioned by region."""
        fs = make_feature_set("value_int__count_groupby", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__count_groupby")
        assert result_col == EXPECTED_COUNT_BY_REGION

    def test_min_groupby_region(self) -> None:
        """Minimum of value_int partitioned by region."""
        fs = make_feature_set("value_int__min_groupby", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__min_groupby")
        assert result_col == EXPECTED_MIN_BY_REGION

    def test_max_groupby_region(self) -> None:
        """Maximum of value_int partitioned by region."""
        fs = make_feature_set("value_int__max_groupby", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "value_int__max_groupby")
        assert result_col == EXPECTED_MAX_BY_REGION

    def test_ec016_avg_with_null_values_in_group(self) -> None:
        """EC-016: Group B has a null value_int at row 4. Avg should skip it."""
        fs = make_feature_set("value_int__avg_groupby", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__avg_groupby")
        assert result_col[4] == pytest.approx(GROUP_B_AVG_EXPECTED, rel=1e-6)
        assert result_col[5] == pytest.approx(GROUP_B_AVG_EXPECTED, rel=1e-6)
        assert result_col[6] == pytest.approx(GROUP_B_AVG_EXPECTED, rel=1e-6)
        assert result_col[7] == pytest.approx(GROUP_B_AVG_EXPECTED, rel=1e-6)

    def test_ec019_null_group_key_forms_own_group(self) -> None:
        """EC-019: Row 11 has region=None. It should form its own group."""
        fs = make_feature_set("value_int__sum_groupby", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "value_int__sum_groupby")
        assert result_col[11] == NULL_GROUP_SUM_EXPECTED

    def test_output_rows_equal_input_rows(self) -> None:
        """Output must have exactly 12 rows, same as input."""
        fs = make_feature_set("value_int__sum_groupby", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert self.get_row_count(result) == 12

    def test_new_column_added(self) -> None:
        """The aggregation result column should be added to the output."""
        fs = make_feature_set("value_int__max_groupby", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_cols = self.extract_column(result, "value_int__max_groupby")
        assert len(result_cols) == 12

    def test_result_has_correct_type(self) -> None:
        """The result of calculate_feature must be the expected framework type."""
        fs = make_feature_set("value_int__min_groupby", ["region"])
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())

    # -- Cross-framework comparison (matches PyArrow reference) --------------

    def _compare_with_pyarrow(self, feature_name: str, partition_by: list[str], use_approx: bool = False) -> None:
        """Run the feature through this framework and PyArrow, assert results match."""
        fs = make_feature_set(feature_name, partition_by)
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

    def test_cross_framework_sum(self) -> None:
        """Sum must match PyArrow reference."""
        self._compare_with_pyarrow("value_int__sum_groupby", ["region"])

    def test_cross_framework_avg(self) -> None:
        """Avg must match PyArrow reference."""
        self._compare_with_pyarrow("value_int__avg_groupby", ["region"], use_approx=True)

    def test_cross_framework_count(self) -> None:
        """Count must match PyArrow reference."""
        self._compare_with_pyarrow("value_int__count_groupby", ["region"])

    def test_cross_framework_min(self) -> None:
        """Min must match PyArrow reference."""
        self._compare_with_pyarrow("value_int__min_groupby", ["region"])

    def test_cross_framework_max(self) -> None:
        """Max must match PyArrow reference."""
        self._compare_with_pyarrow("value_int__max_groupby", ["region"])
