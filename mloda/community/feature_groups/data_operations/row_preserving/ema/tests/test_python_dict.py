"""Tests for PythonDictEma compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.ema.python_dict_ema import (
    PythonDictEma,
)
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.ema.ema import (
    EmaTestBase,
)


class TestPythonDictEma(PythonDictTestMixin, EmaTestBase):
    """All value/semantics/error tests inherited from the base class.

    No overrides expected: PythonDict, being pure Python, can implement the
    exact pinned recurrence natively (unlike pyarrow/duckdb/sqlite, which have
    no exponentially-weighted primitive and ship no backend at all). PythonDict
    targets the same full support level as pandas and polars-lazy.
    """

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictEma
