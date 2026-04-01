"""Tests for StringFeatureGroup base class."""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.match_validation import MatchValidationTestBase
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.string.base import (
    STRING_OPS,
    StringFeatureGroup,
)


class TestStringOpsRegistry:
    def test_string_ops_contains_all_operations(self) -> None:
        expected_ops = {"upper", "lower", "trim", "length", "reverse"}
        for op in expected_ops:
            assert op in STRING_OPS, f"Missing operation: {op}"


class TestPatternMatching:
    @pytest.mark.parametrize(
        "feature_name",
        [
            "name__upper",
            "name__lower",
            "name__trim",
            "name__length",
            "name__reverse",
        ],
    )
    def test_matches_all_operations(self, feature_name: str) -> None:
        options = Options()
        result = StringFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is True, f"Should match: {feature_name}"

    def test_no_match_wrong_suffix(self) -> None:
        options = Options()
        result = StringFeatureGroup.match_feature_group_criteria("name__split", options, None)
        assert result is False

    def test_no_match_no_suffix(self) -> None:
        options = Options()
        result = StringFeatureGroup.match_feature_group_criteria("name", options, None)
        assert result is False

    def test_no_match_no_source_column(self) -> None:
        options = Options()
        result = StringFeatureGroup.match_feature_group_criteria("upper", options, None)
        assert result is False

    def test_no_match_empty_prefix(self) -> None:
        """Empty prefix before __ must not match (requires at least one character)."""
        options = Options()
        result = StringFeatureGroup.match_feature_group_criteria("__upper", options, None)
        assert result is False

    def test_multi_underscore_source_column(self) -> None:
        """Source columns with underscores (e.g. first_name) should match."""
        options = Options()
        result = StringFeatureGroup.match_feature_group_criteria("first_name__upper", options, None)
        assert result is True


class TestPatternParsing:
    def test_parse_upper_operation(self) -> None:
        operation = StringFeatureGroup.get_string_op("name__upper")
        assert operation == "upper"

    def test_parse_length_operation(self) -> None:
        operation = StringFeatureGroup.get_string_op("title__length")
        assert operation == "length"

    def test_parse_source_feature(self) -> None:
        feature = Feature("name__upper", options=Options())
        source_features = StringFeatureGroup._extract_source_features(feature)
        assert source_features == ["name"]

    def test_parse_source_feature_with_underscores(self) -> None:
        feature = Feature("first_name__lower", options=Options())
        source_features = StringFeatureGroup._extract_source_features(feature)
        assert source_features == ["first_name"]

    def test_get_string_op_raises_for_invalid_name(self) -> None:
        """get_string_op should raise ValueError for names without a valid operation."""
        with pytest.raises(ValueError, match="Could not extract string operation"):
            StringFeatureGroup.get_string_op("name__capitalize")

    def test_extract_string_op_config_based(self) -> None:
        """_extract_string_op should fall back to options when name has no pattern."""
        feature = Feature(
            "my_result",
            options=Options(context={"string_op": "lower", "in_features": "name"}),
        )
        op = StringFeatureGroup._extract_string_op(feature)
        assert op == "lower"

    def test_extract_string_op_raises_when_missing(self) -> None:
        """_extract_string_op should raise when neither name nor options contain the op."""
        feature = Feature("my_result", options=Options())
        with pytest.raises(ValueError, match="Could not extract string operation"):
            StringFeatureGroup._extract_string_op(feature)


class TestConfigBasedFeatures:
    def test_config_based_match(self) -> None:
        options = Options(
            context={
                "string_op": "upper",
                "in_features": "name",
            }
        )
        result = StringFeatureGroup.match_feature_group_criteria("my_upper_result", options, None)
        assert result is True

    def test_config_based_match_rejects_invalid_op(self) -> None:
        options = Options(
            context={
                "string_op": "invalid_op",
                "in_features": "name",
            }
        )
        result = StringFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False


class TestValidateStringMatch:
    def test_base_class_validates_known_ops(self) -> None:
        """Base class _validate_string_match accepts all STRING_OPS."""
        for op in STRING_OPS:
            assert StringFeatureGroup._validate_string_match(f"name__{op}", op, "name") is True

    def test_base_class_rejects_unknown_ops(self) -> None:
        """Base class _validate_string_match rejects operations not in STRING_OPS."""
        assert StringFeatureGroup._validate_string_match("name__capitalize", "capitalize", "name") is False


class TestStringMatchValidation(MatchValidationTestBase):
    @classmethod
    def feature_group_class(cls) -> Any:
        return StringFeatureGroup

    @classmethod
    def valid_operations(cls) -> set[str]:
        return set(STRING_OPS)

    @classmethod
    def config_key(cls) -> str:
        return "string_op"

    @classmethod
    def build_feature_name(cls, operation: str) -> str:
        return f"name__{operation}"

    @classmethod
    def build_feature_name_no_source(cls) -> str:
        return "upper"

    @classmethod
    def additional_match_options(cls) -> dict[str, Any]:
        return {"in_features": "name"}
