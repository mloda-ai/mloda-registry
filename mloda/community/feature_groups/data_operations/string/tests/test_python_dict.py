"""Tests for PythonDictStringOps compute implementation."""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from mloda.community.feature_groups.data_operations.string.pyarrow_string import (
    PyArrowStringOps,
)
from mloda.community.feature_groups.data_operations.string.python_dict_string import (
    PythonDictStringOps,
)
from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.string.string import (
    StringTestBase,
)


class TestPythonDictStringOps(PythonDictTestMixin, StringTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictStringOps

    # -- Unicode full vs. simple case-mapping divergence --------------------
    #
    # Python's own ``str.upper()`` / ``str.lower()`` use "full" Unicode case
    # mapping, which applies context-sensitive special-casing rules (e.g.
    # expanding German sharp s, adding a combining dot to Turkish dotted I).
    # The PyArrow reference (``pc.utf8_upper`` / ``pc.utf8_lower``) uses
    # "simple", context-free, per-codepoint case mapping instead, and does
    # not apply these expansions. ``PythonDictStringOps._upper`` / ``_lower``
    # consult the ``_UPPER_OVERRIDES`` / ``_LOWER_OVERRIDES`` tables first
    # (the closed set of codepoints where full and simple mapping disagree)
    # and only fall back to Python's per-character ``.upper()`` / ``.lower()``
    # for every other codepoint, reproducing PyArrow's simple mapping.
    #
    # Empirically, pandas' current default string dtype (``ArrowStringArray``,
    # backed by PyArrow storage) delegates ``.str.upper()`` / ``.str.lower()``
    # to PyArrow's simple case mapping and therefore matches the PyArrow
    # reference on both inputs below, so this is not an already-shared,
    # pre-existing divergence: it is specific to the current PythonDictStringOps
    # implementation.

    def test_upper_diverges_from_pyarrow_on_german_sharp_s(self) -> None:
        """German sharp s (``ß``, U+00DF) uppercases to ``SS`` under Python's
        full Unicode case mapping, but to ``ẞ`` (U+1E9E, capital sharp s)
        under PyArrow's simple case mapping. PythonDictStringOps must match
        the PyArrow reference (and current pandas default behavior), not
        Python's own ``str.upper()``."""
        table = pa.table({"name": ["straße"]})
        data = self.create_test_data(table)
        fs = make_feature_set("name__upper")

        result = self.implementation_class().calculate_feature(data, fs)
        result_col = self.extract_column(result, "name__upper")

        reference = PyArrowStringOps().calculate_feature(table, fs)
        reference_col = extract_column(reference, "name__upper")

        assert result_col == reference_col == ["STRAẞE"], (
            f"expected PyArrow's simple case mapping {reference_col!r}, got PythonDictStringOps result {result_col!r}"
        )

    def test_lower_diverges_from_pyarrow_on_turkish_dotted_i(self) -> None:
        """Turkish dotted capital I (``İ``, U+0130) lowercases to ``i`` plus a
        combining dot above (``i̇``) under Python's full Unicode case
        mapping, but to plain ``i`` under PyArrow's simple case mapping.
        PythonDictStringOps must match the PyArrow reference (and current
        pandas default behavior), not Python's own ``str.lower()``."""
        table = pa.table({"name": ["İstanbul"]})
        data = self.create_test_data(table)
        fs = make_feature_set("name__lower")

        result = self.implementation_class().calculate_feature(data, fs)
        result_col = self.extract_column(result, "name__lower")

        reference = PyArrowStringOps().calculate_feature(table, fs)
        reference_col = extract_column(reference, "name__lower")

        assert result_col == reference_col == ["istanbul"], (
            f"expected PyArrow's simple case mapping {reference_col!r}, got PythonDictStringOps result {result_col!r}"
        )


class TestPythonDictStringCaseMappingDriftCheck:
    """``_UPPER_OVERRIDES`` / ``_LOWER_OVERRIDES`` are a hand-pasted snapshot of one
    PyArrow build's Unicode special-casing data, verified against only two codepoints by
    the tests above. Scanning every codepoint against a live PyArrow build here re-derives
    the invariant on every run, so a future utf8proc bump that adds, changes, or removes a
    special-casing entry fails a test instead of drifting silently.
    """

    @staticmethod
    def _all_codepoints() -> list[str]:
        return [chr(cp) for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]

    def test_upper_matches_pyarrow_for_every_codepoint(self) -> None:
        chars = self._all_codepoints()
        table = pa.table({"name": chars})
        fs = make_feature_set("name__upper")

        data = {"name": list(chars)}
        result = PythonDictStringOps.calculate_feature(data, fs)
        result_col = extract_column(result, "name__upper")

        reference = PyArrowStringOps().calculate_feature(table, fs)
        reference_col = extract_column(reference, "name__upper")

        mismatches = [
            (hex(ord(ch)), ch, ours, ref) for ch, ours, ref in zip(chars, result_col, reference_col) if ours != ref
        ]
        assert not mismatches, (
            f"{len(mismatches)} codepoint(s) disagree with PyArrow utf8_upper (first 10): {mismatches[:10]!r}"
        )

    def test_lower_matches_pyarrow_for_every_codepoint(self) -> None:
        chars = self._all_codepoints()
        table = pa.table({"name": chars})
        fs = make_feature_set("name__lower")

        data = {"name": list(chars)}
        result = PythonDictStringOps.calculate_feature(data, fs)
        result_col = extract_column(result, "name__lower")

        reference = PyArrowStringOps().calculate_feature(table, fs)
        reference_col = extract_column(reference, "name__lower")

        mismatches = [
            (hex(ord(ch)), ch, ours, ref) for ch, ours, ref in zip(chars, result_col, reference_col) if ours != ref
        ]
        assert not mismatches, (
            f"{len(mismatches)} codepoint(s) disagree with PyArrow utf8_lower (first 10): {mismatches[:10]!r}"
        )
