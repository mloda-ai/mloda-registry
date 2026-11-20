"""Tests for PythonDictDateTimeExtraction compute implementation."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pytest
from mloda.user import DataType

from mloda.community.feature_groups.data_operations.row_preserving.datetime.python_dict_datetime import (
    PythonDictDateTimeExtraction,
)
from mloda.testing.feature_groups.data_operations.mixins.frameworks import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.datetime.datetime import (
    DateTimeTestBase,
)
from mloda.testing.typed_data import TypedTestData


class TestPythonDictDateTimeExtraction(PythonDictTestMixin, DateTimeTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictDateTimeExtraction

    # -- Non-timestamp source column guard -----------------------------------
    #
    # Date-level ops accept bare dates; time-of-day ops on dates and string sources raise ValueError.

    def test_date_only_source_hour_raises_value_error(self) -> None:
        """A ``date``-only source column (no time component) must be
        rejected with a ``ValueError``, not silently return ``0`` for the
        ``hour`` op."""
        data = self.create_test_data(TypedTestData({"d": (DataType.DATE, [date(2023, 1, 1), date(2023, 1, 2), None])}))

        with pytest.raises(ValueError, match=r"(?i)date|datetime|timestamp"):
            self.implementation_class()._compute_datetime(data, "d__hour", "d", "hour")

    def test_non_datetime_source_year_raises_value_error(self) -> None:
        """A source column holding a genuinely wrong type (plain strings)
        must be rejected with a ``ValueError``, not an ``AttributeError``
        from accessing ``.year`` on a non-datetime object."""
        data = self.create_test_data(TypedTestData({"d": (DataType.STRING, ["not-a-date", "also-not-a-date", None])}))

        with pytest.raises(ValueError, match=r"(?i)date|datetime|timestamp"):
            self.implementation_class()._compute_datetime(data, "d__year", "d", "year")

    def test_mixed_datetime_and_date_source_hour_raises_value_error(self) -> None:
        """A column holding a datetime and a bare date is rejected as a whole for ``hour``."""
        data = {"d": [datetime(2023, 1, 1, 5, 0, 0), date(2023, 1, 2), None]}

        with pytest.raises(ValueError, match=r"(?i)date|datetime|timestamp"):
            self.implementation_class()._compute_datetime(data, "d__hour", "d", "hour")
