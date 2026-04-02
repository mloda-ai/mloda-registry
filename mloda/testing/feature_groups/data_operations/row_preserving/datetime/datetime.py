"""Shared test base class, data, and helpers for datetime extraction tests.

Expected values are computed from the canonical 12-row dataset in
DataOperationsTestDataCreator, using the 'timestamp' column.

Timestamps (UTC):
  Row 0:  2023-01-01 00:00:00 (Sun)   Row 6:  2023-01-07 00:00:00 (Sat)
  Row 1:  2023-01-02 00:00:00 (Mon)   Row 7:  2023-01-08 00:00:00 (Sun)
  Row 2:  2023-01-03 00:00:00 (Tue)   Row 8:  2023-01-09 00:00:00 (Mon)
  Row 3:  2023-01-05 00:00:00 (Thu)   Row 9:  2023-01-10 00:00:00 (Tue)
  Row 4:  2023-01-06 00:00:00 (Fri)   Row 10: None (null)
  Row 5:  2023-01-06 00:00:00 (Fri)   Row 11: 2023-01-12 00:00:00 (Thu)

A supplementary 4-row dataset (``_varied_times_arrow_table``) adds
non-midnight timestamps to exercise non-zero hour, minute, and second
extraction:

  Row 0: 2023-06-15 14:30:45 (Thu)
  Row 1: 2023-09-22 08:15:30 (Fri)
  Row 2: 2023-12-31 23:59:59 (Sun)
  Row 3: None (null)

The ``DateTimeTestBase`` class provides concrete test methods for all 9
datetime operations (year, month, day, hour, minute, second, dayofweek,
is_weekend, quarter) plus null handling, row preservation, type checks,
cross-framework comparison, and non-midnight time extraction tests.
Framework implementations inherit all tests by subclassing and implementing
a few abstract methods.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pyarrow as pa
import pytest

from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set


# ---------------------------------------------------------------------------
# Expected values (module-level constants, canonical 12-row dataset)
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
# Supplementary dataset: non-midnight timestamps for hour/minute/second
# ---------------------------------------------------------------------------

_VARIED_TIMES_TIMESTAMPS = [
    datetime(2023, 6, 15, 14, 30, 45, tzinfo=timezone.utc),  # Thu, Q2
    datetime(2023, 9, 22, 8, 15, 30, tzinfo=timezone.utc),  # Fri, Q3
    datetime(2023, 12, 31, 23, 59, 59, tzinfo=timezone.utc),  # Sun, Q4
    None,  # null
]

VARIED_EXPECTED_YEAR: list[Any] = [2023, 2023, 2023, None]
VARIED_EXPECTED_MONTH: list[Any] = [6, 9, 12, None]
VARIED_EXPECTED_DAY: list[Any] = [15, 22, 31, None]
VARIED_EXPECTED_HOUR: list[Any] = [14, 8, 23, None]
VARIED_EXPECTED_MINUTE: list[Any] = [30, 15, 59, None]
VARIED_EXPECTED_SECOND: list[Any] = [45, 30, 59, None]
# Thu=3, Fri=4, Sun=6, None
VARIED_EXPECTED_DAYOFWEEK: list[Any] = [3, 4, 6, None]
# Thu=0, Fri=0, Sun=1, None
VARIED_EXPECTED_IS_WEEKEND: list[Any] = [0, 0, 1, None]
# Q2, Q3, Q4, None
VARIED_EXPECTED_QUARTER: list[Any] = [2, 3, 4, None]

VARIED_EXPECTED: dict[str, list[Any]] = {
    "year": VARIED_EXPECTED_YEAR,
    "month": VARIED_EXPECTED_MONTH,
    "day": VARIED_EXPECTED_DAY,
    "hour": VARIED_EXPECTED_HOUR,
    "minute": VARIED_EXPECTED_MINUTE,
    "second": VARIED_EXPECTED_SECOND,
    "dayofweek": VARIED_EXPECTED_DAYOFWEEK,
    "is_weekend": VARIED_EXPECTED_IS_WEEKEND,
    "quarter": VARIED_EXPECTED_QUARTER,
}


def _create_varied_times_arrow_table() -> pa.Table:
    """Create a 4-row PyArrow table with non-midnight UTC timestamps."""
    return pa.table(
        {
            "timestamp": pa.array(
                _VARIED_TIMES_TIMESTAMPS,
                type=pa.timestamp("us", tz="UTC"),
            ),
        }
    )


# ---------------------------------------------------------------------------
# Reusable test base class
# ---------------------------------------------------------------------------


class DateTimeTestBase(DataOpsTestBase):
    """Abstract base class for datetime extraction framework tests."""

    @classmethod
    def reference_implementation_class(cls) -> Any:
        from mloda.community.feature_groups.data_operations.row_preserving.datetime.pyarrow_datetime import (
            PyArrowDateTimeExtraction,
        )

        return PyArrowDateTimeExtraction

    # -- Setup (extended for supplementary dataset) --------------------------

    def setup_method(self) -> None:
        super().setup_method()
        self._varied_arrow_table = _create_varied_times_arrow_table()
        self._varied_test_data = self.create_test_data(self._varied_arrow_table)

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

    # -- Non-midnight time extraction tests ----------------------------------

    def test_varied_hour_extraction(self) -> None:
        """Extract non-zero hours from the supplementary dataset."""
        fs = make_feature_set("timestamp__hour")
        result = self.implementation_class().calculate_feature(self._varied_test_data, fs)
        result_col = self.extract_column(result, "timestamp__hour")
        assert result_col == VARIED_EXPECTED_HOUR

    def test_varied_minute_extraction(self) -> None:
        """Extract non-zero minutes from the supplementary dataset."""
        fs = make_feature_set("timestamp__minute")
        result = self.implementation_class().calculate_feature(self._varied_test_data, fs)
        result_col = self.extract_column(result, "timestamp__minute")
        assert result_col == VARIED_EXPECTED_MINUTE

    def test_varied_second_extraction(self) -> None:
        """Extract non-zero seconds from the supplementary dataset."""
        fs = make_feature_set("timestamp__second")
        result = self.implementation_class().calculate_feature(self._varied_test_data, fs)
        result_col = self.extract_column(result, "timestamp__second")
        assert result_col == VARIED_EXPECTED_SECOND

    def test_varied_quarter_extraction(self) -> None:
        """Extract quarters 2, 3, 4 from the supplementary dataset."""
        fs = make_feature_set("timestamp__quarter")
        result = self.implementation_class().calculate_feature(self._varied_test_data, fs)
        result_col = self.extract_column(result, "timestamp__quarter")
        assert result_col == VARIED_EXPECTED_QUARTER

    def test_varied_month_extraction(self) -> None:
        """Extract months 6, 9, 12 from the supplementary dataset."""
        fs = make_feature_set("timestamp__month")
        result = self.implementation_class().calculate_feature(self._varied_test_data, fs)
        result_col = self.extract_column(result, "timestamp__month")
        assert result_col == VARIED_EXPECTED_MONTH

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

    # -- Cross-framework comparison (matches reference) --------------

    def _compare_varied_with_reference(self, feature_name: str) -> None:
        """Run the feature on the supplementary dataset, compare with the reference."""
        fs = make_feature_set(feature_name)
        result = self.implementation_class().calculate_feature(self._varied_test_data, fs)
        ref = self.reference_implementation_class().calculate_feature(self._varied_arrow_table, fs)

        result_col = self.extract_column(result, feature_name)
        ref_col = extract_column(ref, feature_name)

        assert len(result_col) == len(ref_col), f"row count {len(result_col)} != reference {len(ref_col)}"
        assert result_col == ref_col

    def test_cross_framework_year(self) -> None:
        """Year must match reference."""
        self._compare_with_reference("timestamp__year")

    def test_cross_framework_month(self) -> None:
        """Month must match reference."""
        self._compare_with_reference("timestamp__month")

    def test_cross_framework_day(self) -> None:
        """Day must match reference."""
        self._compare_with_reference("timestamp__day")

    def test_cross_framework_hour(self) -> None:
        """Hour must match reference."""
        self._compare_with_reference("timestamp__hour")

    def test_cross_framework_minute(self) -> None:
        """Minute must match reference."""
        self._compare_with_reference("timestamp__minute")

    def test_cross_framework_second(self) -> None:
        """Second must match reference."""
        self._compare_with_reference("timestamp__second")

    def test_cross_framework_dayofweek(self) -> None:
        """Day of week must match reference."""
        self._compare_with_reference("timestamp__dayofweek")

    def test_cross_framework_is_weekend(self) -> None:
        """Is-weekend must match reference."""
        self._compare_with_reference("timestamp__is_weekend")

    def test_cross_framework_quarter(self) -> None:
        """Quarter must match reference."""
        self._compare_with_reference("timestamp__quarter")

    # -- Cross-framework comparison on varied-times dataset ------------------

    def test_cross_framework_varied_hour(self) -> None:
        """Non-zero hours must match reference on varied dataset."""
        self._compare_varied_with_reference("timestamp__hour")

    def test_cross_framework_varied_minute(self) -> None:
        """Non-zero minutes must match reference on varied dataset."""
        self._compare_varied_with_reference("timestamp__minute")

    def test_cross_framework_varied_second(self) -> None:
        """Non-zero seconds must match reference on varied dataset."""
        self._compare_varied_with_reference("timestamp__second")

    # -- All-null column tests -----------------------------------------------

    def test_all_null_timestamp_column(self) -> None:
        """An all-null timestamp column should produce all None for year extraction."""
        all_null_table = pa.table(
            {
                "timestamp": pa.array([None, None, None], type=pa.timestamp("us", tz="UTC")),
            }
        )
        data = self.create_test_data(all_null_table)
        fs = make_feature_set("timestamp__year")
        result = self.implementation_class().calculate_feature(data, fs)

        result_col = self.extract_column(result, "timestamp__year")
        assert all(v is None for v in result_col), f"expected all None, got {result_col}"

    # -- Option-based config tests -------------------------------------------

    def test_option_based_year(self) -> None:
        """Option-based configuration (not string pattern) produces the same result."""
        from mloda.core.abstract_plugins.components.feature_set import FeatureSet
        from mloda.core.abstract_plugins.components.options import Options
        from mloda.user import Feature

        feature = Feature(
            "my_year",
            options=Options(
                context={
                    "datetime_op": "year",
                    "in_features": "timestamp",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "my_year")
        assert result_col == EXPECTED_YEAR

    def test_unsupported_datetime_op_raises(self) -> None:
        """Calling _compute_datetime with an unknown operation should raise ValueError."""
        with pytest.raises(ValueError, match="[Uu]nsupported"):
            self.implementation_class()._compute_datetime(self.test_data, "timestamp__evil_op", "timestamp", "evil_op")
