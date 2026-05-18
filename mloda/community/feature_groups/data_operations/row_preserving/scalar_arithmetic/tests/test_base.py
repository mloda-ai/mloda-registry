"""Tests for ScalarArithmeticFeatureGroup base class."""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.match_validation import MatchValidationTestBase
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
    ARITHMETIC_OPERATIONS,
    ScalarArithmeticFeatureGroup,
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
        # CONSTANT has strict_validation=False, so missing-constant does not block match.
        options = Options()
        result = ScalarArithmeticFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is True, f"Should match: {feature_name}"

    def test_no_match_wrong_suffix(self) -> None:
        options = Options()
        result = ScalarArithmeticFeatureGroup.match_feature_group_criteria("value_int__add_scalar", options, None)
        assert result is False

    def test_no_match_no_suffix(self) -> None:
        options = Options()
        result = ScalarArithmeticFeatureGroup.match_feature_group_criteria("value_int__add", options, None)
        assert result is False

    def test_no_match_no_source_column(self) -> None:
        options = Options()
        result = ScalarArithmeticFeatureGroup.match_feature_group_criteria("add_constant", options, None)
        assert result is False

    def test_no_match_invalid_operation(self) -> None:
        options = Options()
        result = ScalarArithmeticFeatureGroup.match_feature_group_criteria("value_int__unknown_constant", options, None)
        assert result is False


class TestPatternParsing:
    def test_parse_add_operation(self) -> None:
        operation = ScalarArithmeticFeatureGroup.get_arithmetic_op("value_int__add_constant")
        assert operation == "add"

    def test_parse_subtract_operation(self) -> None:
        operation = ScalarArithmeticFeatureGroup.get_arithmetic_op("value_int__subtract_constant")
        assert operation == "subtract"

    def test_parse_multiply_operation(self) -> None:
        operation = ScalarArithmeticFeatureGroup.get_arithmetic_op("value_int__multiply_constant")
        assert operation == "multiply"

    def test_parse_divide_operation(self) -> None:
        operation = ScalarArithmeticFeatureGroup.get_arithmetic_op("value_int__divide_constant")
        assert operation == "divide"

    def test_parse_source_feature(self) -> None:
        feature = Feature("value_int__add_constant", options=Options(context={"constant": 5}))
        source_features = ScalarArithmeticFeatureGroup._extract_source_features(feature)
        assert source_features == ["value_int"]

    def test_parse_source_feature_with_underscores(self) -> None:
        feature = Feature("my_value__multiply_constant", options=Options(context={"constant": 2}))
        source_features = ScalarArithmeticFeatureGroup._extract_source_features(feature)
        assert source_features == ["my_value"]


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
