"""Tests for PyArrowStringOps compute implementation."""

from __future__ import annotations

import unicodedata
from typing import Any

import pyarrow as pa
import pytest

from mloda.community.feature_groups.data_operations.string.base import StringFeatureGroup
from mloda.community.feature_groups.data_operations.string.pyarrow_string import (
    PyArrowStringOps,
)
from mloda.testing.feature_groups.data_operations.helpers import calculate_with, make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.frameworks import PyArrowTestMixin
from mloda.testing.feature_groups.data_operations.string.reference import reference_string
from mloda.testing.feature_groups.data_operations.string.string import (
    StringTestBase,
)


class TestPyArrowStringOps(PyArrowTestMixin, StringTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PyArrowStringOps


class TestReferenceCaseMappingDriftCheck:
    """The kit reference's upper/lower equal PyArrow over every assigned codepoint."""

    @staticmethod
    def _assigned_codepoints() -> list[str]:
        chars = (chr(cp) for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF)
        return [ch for ch in chars if unicodedata.category(ch) != "Cn"]

    @pytest.mark.parametrize("operation", ["upper", "lower"])
    def test_reference_matches_pyarrow_for_every_codepoint(self, operation: str) -> None:
        chars = self._assigned_codepoints()
        fs = make_feature_set(f"name__{operation}")
        table = pa.table({"name": chars})

        ours = reference_string(StringFeatureGroup, table, fs).column(f"name__{operation}").to_pylist()
        expected = calculate_with(PyArrowStringOps, table, fs).column(f"name__{operation}").to_pylist()

        # Skip mappings the interpreter's Unicode database cannot know (unassigned in this Python).
        mismatches = [
            (hex(ord(ch)), ch, a, b)
            for ch, a, b in zip(chars, ours, expected)
            if a != b and not any(unicodedata.category(c) == "Cn" for c in (b or ""))
        ]
        assert not mismatches, f"{len(mismatches)} codepoint(s) differ from PyArrow (first 10): {mismatches[:10]!r}"
