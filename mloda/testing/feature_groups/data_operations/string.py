"""Shared test base class, data, and helpers for string operation tests.

Expected values are computed from the canonical 12-row dataset in
DataOperationsTestDataCreator, using the 'name' column.

Name values:
  Row 0:  "Alice"    Row 6:  "Grace"
  Row 1:  "bob"      Row 7:  "alice"
  Row 2:  None       Row 8:  "  "      (two spaces)
  Row 3:  ""         Row 9:  "Bob"
  Row 4:  " Eve "    Row 10: "hello"   (with accent e)
  Row 5:  "FRANK"    Row 11: None

The ``StringTestBase`` class provides concrete test methods for all 5
string operations (upper, lower, trim, length, reverse) plus null handling,
empty string handling, row preservation, type checks, and cross-framework
comparison. Framework implementations inherit all tests by subclassing
and implementing a few abstract methods.
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

# Null rows: 2 and 11 (None in source). Row 3 is "" (empty string, not null).
EXPECTED_UPPER: list[Any] = [
    "ALICE",
    "BOB",
    None,
    "",
    " EVE ",
    "FRANK",
    "GRACE",
    "ALICE",
    "  ",
    "BOB",
    "H\u00c9LLO",
    None,
]
EXPECTED_LOWER: list[Any] = [
    "alice",
    "bob",
    None,
    "",
    " eve ",
    "frank",
    "grace",
    "alice",
    "  ",
    "bob",
    "h\u00e9llo",
    None,
]
EXPECTED_TRIM: list[Any] = ["Alice", "bob", None, "", "Eve", "FRANK", "Grace", "alice", "", "Bob", "h\u00e9llo", None]
EXPECTED_LENGTH: list[Any] = [5, 3, None, 0, 5, 5, 5, 5, 2, 3, 5, None]
EXPECTED_REVERSE: list[Any] = [
    "ecilA",
    "bob",
    None,
    "",
    " evE ",
    "KNARF",
    "ecarG",
    "ecila",
    "  ",
    "boB",
    "oll\u00e9h",
    None,
]


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


class StringTestBase(ABC):
    """Abstract base class for string operation framework tests.

    Subclasses implement abstract methods to wire up their framework,
    then inherit concrete test methods for all 5 string operations
    plus null handling, empty string handling, row count checks,
    type checks, and cross-framework comparison against PyArrow.
    """

    ALL_STRING_OPS = {"upper", "lower", "trim", "length", "reverse"}

    @classmethod
    def supported_ops(cls) -> set[str]:
        """String operations this framework supports. Override to restrict."""
        return cls.ALL_STRING_OPS

    # -- Overridable expected values (for frameworks with different behavior) --

    @classmethod
    def expected_upper(cls) -> list[Any]:
        """Override for frameworks where UPPER handles unicode differently."""
        return EXPECTED_UPPER

    @classmethod
    def expected_lower(cls) -> list[Any]:
        """Override for frameworks where LOWER handles unicode differently."""
        return EXPECTED_LOWER

    @classmethod
    def expected_trim(cls) -> list[Any]:
        """Override for frameworks where TRIM handles unicode differently."""
        return EXPECTED_TRIM

    @classmethod
    def expected_length(cls) -> list[Any]:
        """Override for frameworks where LENGTH handles unicode differently."""
        return EXPECTED_LENGTH

    @classmethod
    def expected_reverse(cls) -> list[Any]:
        """Override for frameworks where REVERSE handles unicode differently."""
        return EXPECTED_REVERSE

    # -- Abstract methods subclasses must implement --------------------------

    @classmethod
    @abstractmethod
    def implementation_class(cls) -> Any:
        """Return the StringOps implementation class to test."""

    @classmethod
    def pyarrow_implementation_class(cls) -> Any:
        """Return the PyArrow implementation class (reference for cross-framework comparison)."""
        from mloda.community.feature_groups.data_operations.string.pyarrow_string import (
            PyArrowStringOps,
        )

        return PyArrowStringOps

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

    # -- Concrete test methods: string operations ----------------------------

    def test_upper(self) -> None:
        """Convert name column to uppercase."""
        fs = make_feature_set("name__upper")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "name__upper")
        assert result_col == self.expected_upper()

    def test_lower(self) -> None:
        """Convert name column to lowercase."""
        fs = make_feature_set("name__lower")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        result_col = self.extract_column(result, "name__lower")
        assert result_col == self.expected_lower()

    def test_trim(self) -> None:
        """Strip whitespace from name column."""
        fs = make_feature_set("name__trim")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        result_col = self.extract_column(result, "name__trim")
        assert result_col == self.expected_trim()

    def test_length(self) -> None:
        """Get length of name column values."""
        fs = make_feature_set("name__length")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        result_col = self.extract_column(result, "name__length")
        assert result_col == self.expected_length()

    def _skip_if_unsupported(self, op: str) -> None:
        if op not in self.supported_ops():
            pytest.skip(f"{op} not supported by this framework")

    def test_reverse(self) -> None:
        """Reverse name column values."""
        self._skip_if_unsupported("reverse")
        fs = make_feature_set("name__reverse")
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        result_col = self.extract_column(result, "name__reverse")
        assert result_col == self.expected_reverse()

    # -- Null handling tests -------------------------------------------------

    def test_null_produces_null_for_all_ops(self) -> None:
        """Rows 2 and 11 have null names. All supported string ops should return None."""
        for op in sorted(self.supported_ops()):
            fs = make_feature_set(f"name__{op}")
            result = self.implementation_class().calculate_feature(self.test_data, fs)
            result_col = self.extract_column(result, f"name__{op}")
            assert result_col[2] is None, f"Expected None at row 2 for {op}, got {result_col[2]}"
            assert result_col[11] is None, f"Expected None at row 11 for {op}, got {result_col[11]}"

    # -- Empty string handling tests -----------------------------------------

    def test_empty_string_upper(self) -> None:
        """Row 3 has empty string. Upper of empty string is empty string."""
        fs = make_feature_set("name__upper")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_col = self.extract_column(result, "name__upper")
        assert result_col[3] == ""

    def test_empty_string_length(self) -> None:
        """Row 3 has empty string. Length of empty string is 0."""
        fs = make_feature_set("name__length")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_col = self.extract_column(result, "name__length")
        assert result_col[3] == 0

    def test_whitespace_trim(self) -> None:
        """Row 4 has ' Eve ' and row 8 has '  '. Trim should strip whitespace."""
        fs = make_feature_set("name__trim")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_col = self.extract_column(result, "name__trim")
        assert result_col[4] == "Eve"
        assert result_col[8] == ""

    # -- Row-preserving and type checks --------------------------------------

    def test_output_rows_equal_input_rows(self) -> None:
        """Output must have exactly 12 rows, same as input."""
        fs = make_feature_set("name__upper")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        assert self.get_row_count(result) == 12

    def test_new_column_added(self) -> None:
        """The string result column should be added to the output."""
        fs = make_feature_set("name__lower")
        result = self.implementation_class().calculate_feature(self.test_data, fs)
        result_col = self.extract_column(result, "name__lower")
        assert len(result_col) == 12

    def test_result_has_correct_type(self) -> None:
        """The result of calculate_feature must be the expected framework type."""
        fs = make_feature_set("name__upper")
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

    def test_cross_framework_upper(self) -> None:
        """Upper must match PyArrow reference."""
        self._compare_with_pyarrow("name__upper")

    def test_cross_framework_lower(self) -> None:
        """Lower must match PyArrow reference."""
        self._compare_with_pyarrow("name__lower")

    def test_cross_framework_trim(self) -> None:
        """Trim must match PyArrow reference."""
        self._compare_with_pyarrow("name__trim")

    def test_cross_framework_length(self) -> None:
        """Length must match PyArrow reference."""
        self._compare_with_pyarrow("name__length")

    def test_cross_framework_reverse(self) -> None:
        """Reverse must match PyArrow reference."""
        self._skip_if_unsupported("reverse")
        self._compare_with_pyarrow("name__reverse")

    # -- Unsupported operation error path ------------------------------------

    def test_unsupported_operation_raises(self) -> None:
        """Calling _compute_string with an unknown operation should raise ValueError."""
        with pytest.raises(ValueError, match="[Uu]nsupported string operation"):
            self.implementation_class()._compute_string(self.test_data, "name__capitalize", "name", "capitalize")
