"""Tests for ScalarArithmeticFeatureGroup base class."""

from __future__ import annotations

from typing import Any

import pytest

from mloda.user import Feature, FeatureName, Options
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.helpers import extract_column, feature_set_for
from mloda.testing.feature_groups.data_operations.match_validation import MatchValidationTestBase, TokenCase

from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
    ARITHMETIC_OPERATIONS,
    ScalarArithmeticFeatureGroup,
)
from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.pyarrow_scalar_arithmetic import (
    PyArrowScalarArithmetic,
)


class TestClassAttributes:
    def test_prefix_pattern_exists(self) -> None:
        assert hasattr(ScalarArithmeticFeatureGroup, "PREFIX_PATTERN")
        assert isinstance(ScalarArithmeticFeatureGroup.PREFIX_PATTERN, str)

    def test_arithmetic_operations_contains_all_operations(self) -> None:
        expected_ops = {"add", "subtract", "multiply", "divide"}
        for op in expected_ops:
            assert op in ARITHMETIC_OPERATIONS, f"Missing operation: {op}"

    def test_arithmetic_operations_has_no_extra_operations(self) -> None:
        """Only the four canonical operations are defined."""
        assert set(ARITHMETIC_OPERATIONS.keys()) == {"add", "subtract", "multiply", "divide"}

    def test_min_in_features_is_one(self) -> None:
        assert ScalarArithmeticFeatureGroup.MIN_IN_FEATURES == 1

    def test_max_in_features_is_one(self) -> None:
        assert ScalarArithmeticFeatureGroup.MAX_IN_FEATURES == 1

    def test_arithmetic_op_constant(self) -> None:
        assert ScalarArithmeticFeatureGroup.ARITHMETIC_OP == "arithmetic_op"

    def test_constant_option_key(self) -> None:
        assert ScalarArithmeticFeatureGroup.CONSTANT == "constant"


class TestPatternMatching:
    @pytest.mark.parametrize(
        "feature_name",
        [
            "value_int__add_constant",
            "value_int__subtract_constant",
            "value_int__multiply_constant",
            "value_int__divide_constant",
        ],
    )
    def test_matches_all_operations(self, feature_name: str) -> None:
        # A pattern match skips property validation, so a missing constant does not block the match.
        options = Options()
        result = ScalarArithmeticFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is True, f"Should match: {feature_name}"

    @pytest.mark.parametrize(
        "feature_name",
        [
            pytest.param("value_int__add_scalar", id="wrong_suffix"),
            pytest.param("value_int__add", id="no_suffix"),
            pytest.param("add_constant", id="no_source_column"),
            pytest.param("value_int__unknown_constant", id="invalid_operation"),
        ],
    )
    def test_no_match(self, feature_name: str) -> None:
        options = Options()
        result = ScalarArithmeticFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is False


class TestPatternParsing:
    @pytest.mark.parametrize(
        ("feature_name", "expected"),
        [
            pytest.param("value_int__add_constant", "add", id="add"),
            pytest.param("value_int__subtract_constant", "subtract", id="subtract"),
            pytest.param("value_int__multiply_constant", "multiply", id="multiply"),
            pytest.param("value_int__divide_constant", "divide", id="divide"),
        ],
    )
    def test_parse_operation(self, feature_name: str, expected: str) -> None:
        operation = ScalarArithmeticFeatureGroup.get_arithmetic_op(feature_name)
        assert operation == expected

    def test_parse_source_feature(self) -> None:
        feature = Feature("value_int__add_constant", options=Options(context={"constant": 5}))
        source_features = ScalarArithmeticFeatureGroup._extract_source_features(feature)
        assert source_features == ["value_int"]

    def test_parse_source_feature_with_underscores(self) -> None:
        feature = Feature("my_value__multiply_constant", options=Options(context={"constant": 2}))
        source_features = ScalarArithmeticFeatureGroup._extract_source_features(feature)
        assert source_features == ["my_value"]

    def test_greedy_regex_for_chained_op_tokens(self) -> None:
        """Pin the greedy-parse contract shared with sibling families.

        ``rsplit("__", 1)`` plus the greedy ``.*__([\\w]+)_constant$`` pattern
        means that for a chained name like ``value_int__add__subtract_constant``
        the source is ``value_int__add`` and the captured op token is
        ``subtract``. Scalar aggregate's ``.*__([\\w]+)_scalar$`` regex behaves
        identically, and chained feature names such as
        ``value_int__sum_scalar__add_constant`` rely on this exact split.

        A future regex tightening must be a deliberate decision; this test
        exists to surface any silent change to the parse contract.
        """
        feature = Feature(
            "value_int__add__subtract_constant",
            options=Options(context={"constant": 5}),
        )
        assert ScalarArithmeticFeatureGroup._extract_source_features(feature) == ["value_int__add"]
        assert ScalarArithmeticFeatureGroup._extract_arithmetic_op(feature) == "subtract"


class TestConfigBasedFeatures:
    def test_config_based_match(self) -> None:
        options = Options(
            context={
                "arithmetic_op": "add",
                "in_features": "value_int",
                "constant": 5,
            }
        )
        result = ScalarArithmeticFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is True

    def test_config_based_match_rejects_invalid_op(self) -> None:
        options = Options(
            context={
                "arithmetic_op": "invalid_op",
                "in_features": "value_int",
                "constant": 5,
            }
        )
        result = ScalarArithmeticFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False


class TestConstantExtraction:
    """The ``constant`` check the arity harness cannot express.

    Every arity verdict for ``constant`` (bare, wrapped, multi-element, wrong type,
    bool, absent, and the #339 compute regression) is declared on
    ``TestScalarArithmeticMatchValidation``; only the message a direct extractor call
    raises for a value the matcher would already have rejected stays here.
    """

    def test_extract_constant_raises_for_non_numeric(self) -> None:
        feature = Feature("value_int__add_constant", options=Options(context={"constant": "five"}))
        with pytest.raises(ValueError, match="int or float"):
            ScalarArithmeticFeatureGroup._extract_constant(feature)


class TestRejectionReporting:
    """A wrong-typed constant is a reported strict-validation rejection, not a silent non-match."""

    def test_wrong_typed_constant_reports_a_reason(self) -> None:
        options = Options(
            context={
                "arithmetic_op": "add",
                "in_features": "value_int",
                "constant": "five",
            }
        )
        reason = ScalarArithmeticFeatureGroup._strict_validation_rejection_reason("my_result", options)
        assert reason is not None
        assert "constant" in reason

    def test_valid_constant_reports_nothing(self) -> None:
        options = Options(
            context={
                "arithmetic_op": "add",
                "in_features": "value_int",
                "constant": 5,
            }
        )
        reason = ScalarArithmeticFeatureGroup._strict_validation_rejection_reason("my_result", options)
        assert reason is None

    def test_nested_singleton_constant_reports_a_reason(self) -> None:
        """A nested singleton constant is a reported strict-validation rejection, not a silent non-match."""
        for value in ([[5]], ([5],)):
            options = Options(
                context={
                    "arithmetic_op": "add",
                    "in_features": "value_int",
                    "constant": value,
                }
            )
            assert ScalarArithmeticFeatureGroup.match_feature_group_criteria("my_result", options, None) is False
            reason = ScalarArithmeticFeatureGroup._strict_validation_rejection_reason("my_result", options)
            assert reason is not None
            assert "constant" in reason

    def test_multi_element_constant_reports_an_arity_reason(self) -> None:
        """A multi-element constant stays a non-match. ``constant`` is a strict_validation spec, so
        core's own hook already reports the match_guard rejection (the same message the real match
        pass records via ``_validate_match_guards``); this mixin's own arity-naming diagnostic never
        gets a turn for a strict spec, matching the documented contract that this hook must keep
        producing the same messages the match pass records.
        """
        options = Options(
            context={
                "arithmetic_op": "add",
                "in_features": "value_int",
                "constant": [5, 10],
            }
        )
        assert ScalarArithmeticFeatureGroup.match_feature_group_criteria("my_result", options, None) is False
        reason = ScalarArithmeticFeatureGroup._strict_validation_rejection_reason("my_result", options)
        assert reason is not None
        assert "'constant'" in reason
        assert "rejected by match_guard" in reason


class TestSingleColumnEnforcement:
    """Verify that MAX_IN_FEATURES=1 enforces single-column behavior."""

    def test_max_in_features_is_one(self) -> None:
        assert ScalarArithmeticFeatureGroup.MAX_IN_FEATURES == 1

    def test_input_features_rejects_multiple_option_in_features(self) -> None:
        options = Options(
            context={
                "arithmetic_op": "add",
                "in_features": ["col_a", "col_b"],
                "constant": 5,
            }
        )
        instance = ScalarArithmeticFeatureGroup()
        with pytest.raises(ValueError, match="at most 1"):
            instance.input_features(options, FeatureName("my_result"))

    def test_extract_source_features_rejects_multiple_in_features(self) -> None:
        options = Options(
            context={
                "arithmetic_op": "add",
                "in_features": ["col_a", "col_b"],
                "constant": 5,
            }
        )
        feature = Feature("my_result", options=options)
        with pytest.raises(ValueError, match="at most 1"):
            ScalarArithmeticFeatureGroup._extract_source_features(feature)

    def test_extract_source_features_returns_single_item_for_string_pattern(self) -> None:
        feature = Feature("value_int__multiply_constant", options=Options(context={"constant": 2}))
        source_features = ScalarArithmeticFeatureGroup._extract_source_features(feature)
        assert len(source_features) == 1
        assert source_features == ["value_int"]

    def test_extract_source_features_returns_single_item_for_option_config(self) -> None:
        options = Options(
            context={
                "arithmetic_op": "multiply",
                "in_features": "revenue",
                "constant": 2,
            }
        )
        feature = Feature("my_result", options=options)
        source_features = ScalarArithmeticFeatureGroup._extract_source_features(feature)
        assert len(source_features) == 1
        assert source_features == ["revenue"]

    def test_input_features_returns_single_feature_for_string_pattern(self) -> None:
        options = Options(context={"constant": 5})
        instance = ScalarArithmeticFeatureGroup()
        result = instance.input_features(options, FeatureName("value_int__add_constant"))
        assert result is not None
        assert len(result) == 1
        names = {f.name for f in result}
        assert names == {"value_int"}

    def test_input_features_returns_single_feature_for_option_config(self) -> None:
        options = Options(
            context={
                "arithmetic_op": "add",
                "in_features": "revenue",
                "constant": 5,
            }
        )
        instance = ScalarArithmeticFeatureGroup()
        result = instance.input_features(options, FeatureName("my_add_result"))
        assert result is not None
        assert len(result) == 1
        names = {f.name for f in result}
        assert names == {"revenue"}


class TestArithmeticOpExtraction:
    """Verify arithmetic op extraction from both string and option sources."""

    def test_get_arithmetic_op_raises_for_non_pattern_name(self) -> None:
        with pytest.raises(ValueError, match="Could not extract"):
            ScalarArithmeticFeatureGroup.get_arithmetic_op("plain_name")

    def test_extract_arithmetic_op_from_options(self) -> None:
        options = Options(
            context={
                "arithmetic_op": "divide",
                "in_features": "value_int",
                "constant": 2.0,
            }
        )
        feature = Feature("my_result", options=options)
        op = ScalarArithmeticFeatureGroup._extract_arithmetic_op(feature)
        assert op == "divide"

    def test_extract_arithmetic_op_raises_without_option(self) -> None:
        feature = Feature("plain_name", options=Options())
        with pytest.raises(ValueError, match="Could not extract"):
            ScalarArithmeticFeatureGroup._extract_arithmetic_op(feature)

    @pytest.mark.parametrize("op", list(ARITHMETIC_OPERATIONS.keys()))
    def test_get_arithmetic_op_for_all_ops(self, op: str) -> None:
        feature_name = f"col__{op}_constant"
        result = ScalarArithmeticFeatureGroup.get_arithmetic_op(feature_name)
        assert result == op


class TestScalarArithmeticMatchValidation(MatchValidationTestBase):
    """Shared match-validation tests adapted for scalar arithmetic."""

    @classmethod
    def feature_group_class(cls) -> Any:
        return ScalarArithmeticFeatureGroup

    @classmethod
    def valid_operations(cls) -> set[str]:
        return set(ARITHMETIC_OPERATIONS)

    @classmethod
    def config_key(cls) -> str:
        return "arithmetic_op"

    @classmethod
    def build_feature_name(cls, operation: str) -> str:
        return f"value_int__{operation}_constant"

    @classmethod
    def build_feature_name_no_source(cls) -> str:
        return "add_constant"

    @classmethod
    def additional_match_options(cls) -> dict[str, Any]:
        return {"in_features": "value_int", "constant": 5}

    @classmethod
    def token_cases(cls) -> list[TokenCase]:
        # constant is scalar too: one number, never a bool, and the op cannot run without it.
        return [*super().token_cases(), TokenCase("constant", 5, 10, invalid=(True,), required=True)]

    @classmethod
    def dispatch_values(cls, options: Options) -> list[Any]:
        feature = Feature("my_result", options=options)
        return [
            ScalarArithmeticFeatureGroup._extract_arithmetic_op(feature),
            ScalarArithmeticFeatureGroup._extract_constant(feature),
        ]

    @classmethod
    def compute_values(cls, options: Options) -> list[Any] | None:
        # The #339 regression: constant=(5,) matched at discovery and then raised
        # "must be int or float, got tuple" inside calculate_feature.
        result = PyArrowScalarArithmetic.calculate_feature(
            PyArrowDataOpsTestDataCreator.create(), feature_set_for("my_result", options)
        )
        return extract_column(result, "my_result")
