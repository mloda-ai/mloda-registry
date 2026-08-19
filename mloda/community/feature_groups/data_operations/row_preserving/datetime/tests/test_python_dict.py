"""Tests for PythonDictDateTimeExtraction compute implementation."""

from __future__ import annotations

from datetime import date
from typing import Any

import pyarrow as pa
import pytest

from mloda.community.feature_groups.data_operations.row_preserving.datetime.python_dict_datetime import (
    PythonDictDateTimeExtraction,
)
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.datetime.datetime import (
    DateTimeTestBase,
)


class TestPythonDictDateTimeExtraction(PythonDictTestMixin, DateTimeTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictDateTimeExtraction

    # -- Non-timestamp source column guard -----------------------------------
    #
    # ``PythonDictDateTimeExtraction._compute_datetime`` runs
    # ``_assert_source_column_is_datetime`` up front, so a source column
    # holding ``datetime.date`` objects (no time component) or a genuinely
    # wrong type (e.g. plain strings) is rejected with a clear ``ValueError``
    # before ``_extract`` ever sees the value, per the "CFW Backend Rejection
    # over Python Fallback" rule (see
    # ``sqlite_time_bucketization._assert_source_column_is_timestamp`` for
    # the established rejection-guard precedent). The PyArrow reference
    # (``pyarrow_datetime.py``) does not currently raise ``ValueError`` for
    # either input; it raises ``pyarrow.lib.ArrowNotImplementedError`` (a
    # ``RuntimeError`` subtype) from the underlying compute kernel. These
    # tests therefore pin the desired PythonDict behavior directly rather
    # than diffing against the PyArrow reference's current, also-imperfect,
    # behavior.

    def test_date_only_source_hour_raises_value_error(self) -> None:
        """A ``date``-only source column (no time component) must be
        rejected with a ``ValueError``, not silently return ``0`` for the
        ``hour`` op."""
        table = pa.table({"d": [date(2023, 1, 1), date(2023, 1, 2), None]})
        data = self.create_test_data(table)

        with pytest.raises(ValueError, match=r"(?i)date|datetime|timestamp"):
            self.implementation_class()._compute_datetime(data, "d__hour", "d", "hour")

    def test_non_datetime_source_year_raises_value_error(self) -> None:
        """A source column holding a genuinely wrong type (plain strings)
        must be rejected with a ``ValueError``, not an ``AttributeError``
        from accessing ``.year`` on a non-datetime object."""
        table = pa.table({"d": ["not-a-date", "also-not-a-date", None]})
        data = self.create_test_data(table)

        with pytest.raises(ValueError, match=r"(?i)date|datetime|timestamp"):
            self.implementation_class()._compute_datetime(data, "d__year", "d", "year")
