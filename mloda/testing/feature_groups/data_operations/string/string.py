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

from typing import Any

import pyarrow as pa
import pytest

from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase
from mloda.testing.feature_groups.data_operations.helpers import make_feature_set


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
# Reusable test base class
# ---------------------------------------------------------------------------


class StringTestBase(DataOpsTestBase):
    """Abstract base class for string operation framework tests."""

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

    @classmethod
    def reference_implementation_class(cls) -> Any:
        from mloda.community.feature_groups.data_operations.string.pyarrow_string import (
            PyArrowStringOps,
        )

        return PyArrowStringOps

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

    # -- Cross-framework comparison (matches reference) --------------

    def test_cross_framework_upper(self) -> None:
        """Upper must match reference."""
        self._compare_with_reference("name__upper")

    def test_cross_framework_lower(self) -> None:
        """Lower must match reference."""
        self._compare_with_reference("name__lower")

    def test_cross_framework_trim(self) -> None:
        """Trim must match reference."""
        self._compare_with_reference("name__trim")

    def test_cross_framework_length(self) -> None:
        """Length must match reference."""
        self._compare_with_reference("name__length")

    def test_cross_framework_reverse(self) -> None:
        """Reverse must match reference."""
        self._skip_if_unsupported("reverse")
        self._compare_with_reference("name__reverse")

    # -- Unsupported operation error path ------------------------------------

    def test_unsupported_operation_raises(self) -> None:
        """Calling _compute_string with an unknown operation should raise ValueError."""
        with pytest.raises(ValueError, match="[Uu]nsupported string operation"):
            self.implementation_class()._compute_string(self.test_data, "name__capitalize", "name", "capitalize")

    # -- All-null column tests -----------------------------------------------

    def test_all_null_column_upper(self) -> None:
        """Upper on an all-null column should produce all None."""
        table = pa.table(
            {
                "name": pa.array([None, None, None], type=pa.string()),
            }
        )
        data = self.create_test_data(table)
        fs = make_feature_set("name__upper")
        result = self.implementation_class().calculate_feature(data, fs)

        result_col = self.extract_column(result, "name__upper")
        assert all(v is None for v in result_col), f"expected all None, got {result_col}"

    def test_all_null_column_length(self) -> None:
        """Length on an all-null column should produce all None."""
        table = pa.table(
            {
                "name": pa.array([None, None, None], type=pa.string()),
            }
        )
        data = self.create_test_data(table)
        fs = make_feature_set("name__length")
        result = self.implementation_class().calculate_feature(data, fs)

        result_col = self.extract_column(result, "name__length")
        assert all(v is None for v in result_col), f"expected all None, got {result_col}"

    # -- Option-based config tests -------------------------------------------

    def test_option_based_upper(self) -> None:
        """Option-based configuration (not string pattern) produces the same result."""
        from mloda.core.abstract_plugins.components.feature_set import FeatureSet
        from mloda.core.abstract_plugins.components.options import Options
        from mloda.user import Feature

        feature = Feature(
            "my_upper",
            options=Options(
                context={
                    "string_op": "upper",
                    "in_features": "name",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)
        result = self.implementation_class().calculate_feature(self.test_data, fs)

        result_col = self.extract_column(result, "my_upper")
        assert result_col == self.expected_upper()
