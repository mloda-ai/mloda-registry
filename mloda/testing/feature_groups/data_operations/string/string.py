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

from typing import TYPE_CHECKING, Any, ClassVar

import pytest
from mloda.provider import FeatureSet
from mloda.user import DataType

from mloda.testing.data_creator.base import DataOperationsTestDataCreator
from mloda.testing.feature_groups.data_operations.base import DataOpsTestBase, divergent_subtypes
from mloda.testing.feature_groups.data_operations.helpers import calculate_with, make_feature_set, typed_data_without
from mloda.testing.feature_groups.data_operations.mixins.input_validation import (
    InputValidationCase,
    InputValidationTestMixin,
)
from mloda.testing.feature_groups.data_operations.mixins.output_contract import OutputContractTestMixin
from mloda.testing.feature_groups.data_operations.string.pinning_rows import STRING_PINNING
from mloda.testing.typed_data import TypedTestData

if TYPE_CHECKING:
    import pyarrow as pa

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


class StringTestBase(InputValidationTestMixin, OutputContractTestMixin, DataOpsTestBase):
    """Abstract base class for string operation framework tests."""

    _case_fixtures: ClassVar[dict[str, str]] = {"pinned_row_case": "pinned_row_cases"}

    @classmethod
    def pinned_row_cases(cls) -> dict[str, Any]:
        return {f"{i}-{row.case}-{row.feature}": row for i, row in enumerate(STRING_PINNING.rows)}

    @classmethod
    def _full_case_mapping_declared(cls, op: str) -> bool:
        """Whether the Implementation declares full case mapping for upper/lower."""
        from mloda.provider import Implementation

        impl = cls.implementation_class()
        if not (isinstance(impl, type) and issubclass(impl, Implementation)):
            return False
        return op in ("upper", "lower") and op in divergent_subtypes(impl, named_only=True)

    def test_pinned_row(self, pinned_row_case: str) -> None:
        """Hand-computed rows; a declared upper/lower divergence pins Python's full mapping on case_mapping rows."""
        row = self.pinned_row_cases()[pinned_row_case]
        op = self.feature_group_class().get_string_op(row.feature)
        self._skip_if_unsupported(op)
        expected = list(row.expected[row.feature][1])
        if row.case == "case_mapping" and self._full_case_mapping_declared(op):
            source = next(iter(row.inputs.values()))[1]
            expected = [None if v is None else (v.upper() if op == "upper" else v.lower()) for v in source]
        result = calculate_with(
            self.implementation_class(),
            self.create_test_data(TypedTestData(row.inputs)),
            make_feature_set(row.feature, **row.context),
        )
        assert self.extract_column(result, row.feature) == expected

    # -- OutputContractTestMixin configuration ----------------------------------
    # trim is supported by every framework, unlike upper/lower/reverse.

    def output_contract_feature_set(self) -> FeatureSet:
        return make_feature_set("name__trim")

    ALL_STRING_OPS = {"upper", "lower", "trim", "length", "reverse"}

    # -- InputValidationTestMixin configuration ---------------------------------

    @classmethod
    def input_validation_cases(cls) -> dict[str, Any]:
        return {
            "multi_column_in_features": InputValidationCase(
                "bad_multi",
                {"string_op": "length", "in_features": ["name", "category"]},
                "at most 1",
            ),
            "missing_source_column": InputValidationCase(
                "name__trim",
                {},
                "Source column 'name' is not present",
                data=typed_data_without(DataOperationsTestDataCreator.typed_data(), "name"),
            ),
            "empty_partition_by": None,
            "missing_partition_by_column": None,
            "missing_order_by_column": None,
        }

    @classmethod
    def supported_ops(cls) -> set[str]:
        """Subtypes the Implementation declares; the full set for a stub."""
        return cls.declared_supported(cls.ALL_STRING_OPS)

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
    def reference_calculate(cls, table: pa.Table, features: Any) -> Any:
        """Run the pure Python reference function on the canonical table."""
        from mloda.testing.feature_groups.data_operations.string.reference import reference_string

        return reference_string(cls.feature_group_class(), table, features)

    # -- Concrete test methods: string operations ----------------------------

    def test_upper(self) -> None:
        """Convert name column to uppercase."""
        self._skip_if_unsupported("upper")
        fs = make_feature_set("name__upper")
        result = calculate_with(self.implementation_class(), self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        assert self.get_row_count(result) == 12

        result_col = self.extract_column(result, "name__upper")
        assert result_col == self.expected_upper()

    def test_lower(self) -> None:
        """Convert name column to lowercase."""
        self._skip_if_unsupported("lower")
        fs = make_feature_set("name__lower")
        result = calculate_with(self.implementation_class(), self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        result_col = self.extract_column(result, "name__lower")
        assert result_col == self.expected_lower()

    def test_trim(self) -> None:
        """Strip whitespace from name column."""
        fs = make_feature_set("name__trim")
        result = calculate_with(self.implementation_class(), self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        result_col = self.extract_column(result, "name__trim")
        assert result_col == self.expected_trim()

    def test_length(self) -> None:
        """Get length of name column values."""
        fs = make_feature_set("name__length")
        result = calculate_with(self.implementation_class(), self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        result_col = self.extract_column(result, "name__length")
        assert result_col == self.expected_length()

    def test_reverse(self) -> None:
        """Reverse name column values."""
        self._skip_if_unsupported("reverse")
        fs = make_feature_set("name__reverse")
        result = calculate_with(self.implementation_class(), self.test_data, fs)

        assert isinstance(result, self.get_expected_type())
        result_col = self.extract_column(result, "name__reverse")
        assert result_col == self.expected_reverse()

    # -- Null handling tests -------------------------------------------------

    def test_null_produces_null_for_all_ops(self) -> None:
        """Rows 2 and 11 have null names. All supported string ops should return None."""
        for op in sorted(self.supported_ops()):
            fs = make_feature_set(f"name__{op}")
            result = calculate_with(self.implementation_class(), self.test_data, fs)
            result_col = self.extract_column(result, f"name__{op}")
            assert result_col[2] is None, f"Expected None at row 2 for {op}, got {result_col[2]}"
            assert result_col[11] is None, f"Expected None at row 11 for {op}, got {result_col[11]}"

    # -- Empty string handling tests -----------------------------------------

    def test_empty_string_upper(self) -> None:
        """Row 3 has empty string. Upper of empty string is empty string."""
        self._skip_if_unsupported("upper")
        fs = make_feature_set("name__upper")
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        result_col = self.extract_column(result, "name__upper")
        assert result_col[3] == ""

    def test_empty_string_length(self) -> None:
        """Row 3 has empty string. Length of empty string is 0."""
        fs = make_feature_set("name__length")
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        result_col = self.extract_column(result, "name__length")
        assert result_col[3] == 0

    def test_whitespace_trim(self) -> None:
        """Row 4 has ' Eve ' and row 8 has '  '. Trim should strip whitespace."""
        fs = make_feature_set("name__trim")
        result = calculate_with(self.implementation_class(), self.test_data, fs)
        result_col = self.extract_column(result, "name__trim")
        assert result_col[4] == "Eve"
        assert result_col[8] == ""

    # -- Cross-framework comparison (matches reference) --------------

    @pytest.mark.parametrize(
        ("operation", "may_skip"),
        [
            pytest.param("upper", True, id="upper"),
            pytest.param("lower", True, id="lower"),
            # No framework in this repo excludes trim or length from supported_ops(),
            # so these two cases are not given a skip.
            pytest.param("trim", False, id="trim"),
            pytest.param("length", False, id="length"),
            pytest.param("reverse", True, id="reverse"),
        ],
    )
    def test_cross_framework(self, operation: str, may_skip: bool) -> None:
        if may_skip:
            self._skip_if_unsupported(operation)
        self._compare_with_reference(f"name__{operation}")

    # -- Unsupported operation error path ------------------------------------

    def test_unsupported_operation_raises(self) -> None:
        """Calling _compute_string with an unknown operation should raise ValueError."""
        with pytest.raises(ValueError, match="[Uu]nsupported string operation"):
            self.implementation_class()._compute_string(self.test_data, "name__capitalize", "name", "capitalize")

    # -- All-null column tests -----------------------------------------------

    def test_all_null_column_upper(self) -> None:
        """Upper on an all-null column should produce all None."""
        self._skip_if_unsupported("upper")
        data = self.create_test_data(TypedTestData({"name": (DataType.STRING, [None, None, None])}))
        fs = make_feature_set("name__upper")
        result = calculate_with(self.implementation_class(), data, fs)

        result_col = self.extract_column(result, "name__upper")
        assert all(v is None for v in result_col), f"expected all None, got {result_col}"

    def test_all_null_column_length(self) -> None:
        """Length on an all-null column should produce all None."""
        data = self.create_test_data(TypedTestData({"name": (DataType.STRING, [None, None, None])}))
        fs = make_feature_set("name__length")
        result = calculate_with(self.implementation_class(), data, fs)

        result_col = self.extract_column(result, "name__length")
        assert all(v is None for v in result_col), f"expected all None, got {result_col}"

    # -- Option-based config tests -------------------------------------------

    def test_option_based_upper(self) -> None:
        """Option-based configuration (not string pattern) produces the same result."""
        self._skip_if_unsupported("upper")
        from mloda.provider import FeatureSet
        from mloda.user import Feature, Options

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
        result = calculate_with(self.implementation_class(), self.test_data, fs)

        result_col = self.extract_column(result, "my_upper")
        assert result_col == self.expected_upper()
