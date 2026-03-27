"""Shared test base class, data, and helpers for datetime extraction tests.

Expected values are computed from the canonical 12-row dataset in
DataOperationsTestDataCreator, using the 'timestamp' column.

Timestamps (UTC):
  Row 0:  2023-01-01 (Sun)   Row 6:  2023-01-07 (Sat)
  Row 1:  2023-01-02 (Mon)   Row 7:  2023-01-08 (Sun)
  Row 2:  2023-01-03 (Tue)   Row 8:  2023-01-09 (Mon)
  Row 3:  2023-01-05 (Thu)   Row 9:  2023-01-10 (Tue)
  Row 4:  2023-01-06 (Fri)   Row 10: None (null)
  Row 5:  2023-01-06 (Fri)   Row 11: 2023-01-12 (Thu)

The ``DateTimeTestBase`` class provides concrete test methods for all 9
datetime operations (year, month, day, hour, minute, second, dayofweek,
is_weekend, quarter) plus null handling, row preservation, type checks,
and cross-framework comparison. Framework implementations inherit all
tests by subclassing and implementing a few abstract methods.
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

# All timestamps are in January 2023 => all years = 2023, months = 1, quarters = 1
# Row 10 is null => None for all operations.
EXPECTED_YEAR: list[Any] = [2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, 2023, None, 2023]
EXPECTED_MONTH: list[Any] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, None, 1]
EXPECTED_DAY: list[Any] = [1, 2, 3, 5, 6, 6, 7, 8, 9, 10, None, 12]
EXPECTED_HOUR: list[Any] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, 0]
EXPECTED_MINUTE: list[Any] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, 0]
EXPECTED_SECOND: list[Any] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, 0]

# weekday(): 0=Monday .. 6=Sunday
# Sun=6, Mon=0, Tue=1, Thu=3, Fri=4, Fri=4, Sat=5, Sun=6, Mon=0, Tue=1, None, Thu=3
EXPECTED_DAYOFWEEK: list[Any] = [6, 0, 1, 3, 4, 4, 5, 6, 0, 1, None, 3]

# is_weekend: 1 if weekday >= 5, else 0
# Sun=1, Mon=0, Tue=0, Thu=0, Fri=0, Fri=0, Sat=1, Sun=1, Mon=0, Tue=0, None, Thu=0
EXPECTED_IS_WEEKEND: list[Any] = [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, None, 0]

# All in January => quarter = 1
EXPECTED_QUARTER: list[Any] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, None, 1]


# ---------------------------------------------------------------------------
# Standalone helpers
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


def make_feature_set(feature_name: str) -> FeatureSet:
    """Build a FeatureSet with the given feature name."""
    feature = Feature(feature_name, options=Options())
    fs = FeatureSet()
    fs.add(feature)
    return fs


# ---------------------------------------------------------------------------
# Reusable test base class
# ---------------------------------------------------------------------------


class DateTimeTestBase(ABC):
    """Abstract base class for datetime extraction framework tests.

    Subclasses implement abstract methods to wire up their framework,
    then inherit concrete test methods for all 9 datetime operations
    plus null handling, row count checks, type checks, and
    cross-framework comparison against PyArrow.
    """

    # -- Abstract methods subclasses must implement --------------------------

    @classmethod
    @abstractmethod
    def implementation_class(cls) -> Any:
        """Return the DateTimeExtraction implementation class to test."""

    @classmethod
    def pyarrow_implementation_class(cls) -> Any:
        """Return the PyArrow implementation class (reference for cross-framework comparison)."""
        from mloda.community.feature_groups.data_operations.row_preserving.datetime.pyarrow_datetime import (
            PyArrowDateTimeExtraction,
        )

        return PyArrowDateTimeExtraction

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

    # -- Concrete test methods: datetime operations --------------------------

    def test_year_extraction(self) -> None:
        """Extract year from timestamp column."""
        fs = make_feature_set("timestamp__year")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "timestamp__year")
        assert result_col == EXPECTED_YEAR

    def test_month_extraction(self) -> None:
        """Extract month from timestamp column."""
        fs = make_feature_set("timestamp__month")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        result_col = self.extract_column(result, "timestamp__month")
        assert result_col == EXPECTED_MONTH

    def test_day_extraction(self) -> None:
        """Extract day from timestamp column."""
        fs = make_feature_set("timestamp__day")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        result_col = self.extract_column(result, "timestamp__day")
        assert result_col == EXPECTED_DAY

    def test_hour_extraction(self) -> None:
        """Extract hour from timestamp column."""
        fs = make_feature_set("timestamp__hour")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        result_col = self.extract_column(result, "timestamp__hour")
        assert result_col == EXPECTED_HOUR

    def test_minute_extraction(self) -> None:
        """Extract minute from timestamp column."""
        fs = make_feature_set("timestamp__minute")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        result_col = self.extract_column(result, "timestamp__minute")
        assert result_col == EXPECTED_MINUTE

    def test_second_extraction(self) -> None:
        """Extract second from timestamp column."""
        fs = make_feature_set("timestamp__second")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        result_col = self.extract_column(result, "timestamp__second")
        assert result_col == EXPECTED_SECOND

    def test_dayofweek_extraction(self) -> None:
        """Extract day of week (0=Monday, 6=Sunday) from timestamp column."""
        fs = make_feature_set("timestamp__dayofweek")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        result_col = self.extract_column(result, "timestamp__dayofweek")
        assert result_col == EXPECTED_DAYOFWEEK

    def test_is_weekend_extraction(self) -> None:
        """Extract is_weekend (1 for Sat/Sun, 0 otherwise) from timestamp column."""
        fs = make_feature_set("timestamp__is_weekend")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        result_col = self.extract_column(result, "timestamp__is_weekend")
        assert result_col == EXPECTED_IS_WEEKEND

    def test_quarter_extraction(self) -> None:
        """Extract quarter (1-4) from timestamp column."""
        fs = make_feature_set("timestamp__quarter")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        result_col = self.extract_column(result, "timestamp__quarter")
        assert result_col == EXPECTED_QUARTER

    # -- Null handling tests -------------------------------------------------

    def test_null_timestamp_produces_null(self) -> None:
        """Row 10 has a null timestamp. All datetime ops should return None."""
        for op in ("year", "month", "day", "hour", "minute", "second", "dayofweek", "is_weekend", "quarter"):
            fs = make_feature_set(f"timestamp__{op}")
            result = self.implementation_class().calculate_feature(self.test_data, fs)
            result_col = self.extract_column(result, f"timestamp__{op}")
            assert result_col[10] is None, f"Expected None at row 10 for {op}, got {result_col[10]}"

    # -- Row-preserving and type checks --------------------------------------

    def test_output_rows_equal_input_rows(self) -> None:
        """Output must have exactly 12 rows, same as input."""
        fs = make_feature_set("timestamp__year")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert self.get_row_count(result) == 12

    def test_new_column_added(self) -> None:
        """The extracted column should be added to the output."""
        fs = make_feature_set("timestamp__day")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_col = self.extract_column(result, "timestamp__day")
        assert len(result_col) == 12

    def test_result_has_correct_type(self) -> None:
        """The result of calculate_feature must be the expected framework type."""
        fs = make_feature_set("timestamp__year")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert isinstance(result, self.get_expected_type())

    # -- Cross-framework comparison (matches PyArrow reference) --------------

    def _compare_with_pyarrow(self, feature_name: str) -> None:
        """Run the feature through this framework and PyArrow, assert results match."""
        fs = make_feature_set(feature_name)
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        ref = self.pyarrow_implementation_class().calculate_feature(self._arrow_table, fs)

        result_col = self.extract_column(result, feature_name)
        ref_col = extract_column(ref, feature_name)

        assert len(result_col) == len(ref_col), f"row count {len(result_col)} != reference {len(ref_col)}"
        assert result_col == ref_col

    def test_cross_framework_year(self) -> None:
        """Year must match PyArrow reference."""
        self._compare_with_pyarrow("timestamp__year")

    def test_cross_framework_month(self) -> None:
        """Month must match PyArrow reference."""
        self._compare_with_pyarrow("timestamp__month")

    def test_cross_framework_day(self) -> None:
        """Day must match PyArrow reference."""
        self._compare_with_pyarrow("timestamp__day")

    def test_cross_framework_dayofweek(self) -> None:
        """Day of week must match PyArrow reference."""
        self._compare_with_pyarrow("timestamp__dayofweek")

    def test_cross_framework_is_weekend(self) -> None:
        """Is-weekend must match PyArrow reference."""
        self._compare_with_pyarrow("timestamp__is_weekend")

    def test_cross_framework_quarter(self) -> None:
        """Quarter must match PyArrow reference."""
        self._compare_with_pyarrow("timestamp__quarter")
