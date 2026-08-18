"""Tests for PythonDictOffset compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.offset.python_dict_offset import (
    PythonDictOffset,
)
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.offset.offset import OffsetTestBase


class TestPythonDictOffset(PythonDictTestMixin, OffsetTestBase):
    """All tests inherited from the base class.

    No overrides of ``supported_offset_types()``: PythonDict aims for the same
    full support level as pandas, DuckDB, polars-lazy, and SQLite (all offset
    types, including the parametric lag/lead/diff/pct_change families).
    """

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictOffset
