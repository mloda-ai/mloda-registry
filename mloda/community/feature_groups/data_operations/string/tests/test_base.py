"""Tests for StringFeatureGroup base class."""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.components.options import Options

from mloda.community.feature_groups.data_operations.string.base import (
    STRING_OPS,
    StringFeatureGroup,
)


class TestClassAttributes:
    def test_prefix_pattern_exists(self) -> None:
        assert hasattr(StringFeatureGroup, "PREFIX_PATTERN")
        assert isinstance(StringFeatureGroup.PREFIX_PATTERN, str)

    def test_string_ops_contains_all_operations(self) -> None:
        expected_ops = {"upper", "lower", "trim", "length", "reverse"}
        for op in expected_ops:
            assert op in STRING_OPS, f"Missing operation: {op}"

    def test_min_in_features_is_one(self) -> None:
        assert StringFeatureGroup.MIN_IN_FEATURES == 1

    def test_max_in_features_is_one(self) -> None:
        assert StringFeatureGroup.MAX_IN_FEATURES == 1


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


class TestPatternParsing:
    def test_parse_upper_operation(self) -> None:
        operation = StringFeatureGroup.get_string_op("name__upper")
        assert operation == "upper"

    def test_parse_length_operation(self) -> None:
        operation = StringFeatureGroup.get_string_op("title__length")
        assert operation == "length"

    def test_parse_source_feature(self) -> None:
        from mloda.user import Feature

        feature = Feature("name__upper", options=Options())
        source_features = StringFeatureGroup._extract_source_features(feature)
        assert source_features == ["name"]

    def test_parse_source_feature_with_underscores(self) -> None:
        from mloda.user import Feature

        feature = Feature("first_name__lower", options=Options())
        source_features = StringFeatureGroup._extract_source_features(feature)
        assert source_features == ["first_name"]


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
