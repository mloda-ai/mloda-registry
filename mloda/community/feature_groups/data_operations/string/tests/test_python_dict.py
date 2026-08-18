"""Tests for PythonDictStringOps compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.string.python_dict_string import (
    PythonDictStringOps,
)
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.string.string import (
    StringTestBase,
)


class TestPythonDictStringOps(PythonDictTestMixin, StringTestBase):
    """All tests inherited from the base class.

    PythonDict targets FULL support (5/5 ops), matching PyArrow/Pandas/
    Polars-lazy/DuckDB. Python's own ``str.upper()``/``str.lower()``/
    ``[::-1]`` are Unicode-correct by default, so no ``supported_ops()``
    restriction is needed here (unlike SQLite, whose native UPPER/LOWER
    are ASCII-only and which has no native REVERSE).
    """

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictStringOps
