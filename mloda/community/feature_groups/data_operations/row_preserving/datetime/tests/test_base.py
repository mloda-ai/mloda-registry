"""Tests for DateTimeFeatureGroup base class."""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.match_validation import MatchValidationTestBase

from mloda.community.feature_groups.data_operations.row_preserving.datetime.base import (
    DATETIME_OPS,
    DateTimeFeatureGroup,
)


class TestClassAttributes:
    def test_prefix_pattern_exists(self) -> None:
        assert hasattr(DateTimeFeatureGroup, "PREFIX_PATTERN")
        assert isinstance(DateTimeFeatureGroup.PREFIX_PATTERN, str)

    def test_datetime_ops_contains_all_operations(self) -> None:
        expected_ops = {"year", "month", "day", "hour", "minute", "second", "dayofweek", "is_weekend", "quarter"}
        for op in expected_ops:
            assert op in DATETIME_OPS, f"Missing operation: {op}"

    def test_min_in_features_is_one(self) -> None:
        assert DateTimeFeatureGroup.MIN_IN_FEATURES == 1

    def test_max_in_features_is_one(self) -> None:
        assert DateTimeFeatureGroup.MAX_IN_FEATURES == 1


class TestPatternMatching:
    @pytest.mark.parametrize(
        "feature_name",
        [
            "timestamp__year",
            "timestamp__month",
            "timestamp__day",
            "timestamp__hour",
            "timestamp__minute",
            "timestamp__second",
            "timestamp__dayofweek",
            "timestamp__is_weekend",
            "timestamp__quarter",
        ],
    )
    def test_matches_all_operations(self, feature_name: str) -> None:
        options = Options()
        result = DateTimeFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is True, f"Should match: {feature_name}"

    def test_no_match_wrong_suffix(self) -> None:
        options = Options()
        result = DateTimeFeatureGroup.match_feature_group_criteria("timestamp__weekday", options, None)
        assert result is False

    def test_no_match_no_suffix(self) -> None:
        options = Options()
        result = DateTimeFeatureGroup.match_feature_group_criteria("timestamp", options, None)
        assert result is False

    def test_no_match_no_source_column(self) -> None:
        options = Options()
        result = DateTimeFeatureGroup.match_feature_group_criteria("year", options, None)
        assert result is False


class TestPatternParsing:
    def test_parse_year_operation(self) -> None:
        operation = DateTimeFeatureGroup.get_datetime_op("timestamp__year")
        assert operation == "year"

    def test_parse_is_weekend_operation(self) -> None:
        operation = DateTimeFeatureGroup.get_datetime_op("created_at__is_weekend")
        assert operation == "is_weekend"

    def test_parse_source_feature(self) -> None:
        from mloda.user import Feature

        feature = Feature("timestamp__year", options=Options())
        source_features = DateTimeFeatureGroup._extract_source_features(feature)
        assert source_features == ["timestamp"]

    def test_parse_source_feature_with_underscores(self) -> None:
        from mloda.user import Feature

        feature = Feature("created_at__month", options=Options())
        source_features = DateTimeFeatureGroup._extract_source_features(feature)
        assert source_features == ["created_at"]


class TestConfigBasedFeatures:
    def test_config_based_match(self) -> None:
        options = Options(
            context={
                "datetime_op": "year",
                "in_features": "timestamp",
            }
        )
        result = DateTimeFeatureGroup.match_feature_group_criteria("my_year_result", options, None)
        assert result is True

    def test_config_based_match_rejects_invalid_op(self) -> None:
        options = Options(
            context={
                "datetime_op": "invalid_op",
                "in_features": "timestamp",
            }
        )
        result = DateTimeFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False


class TestDateTimeMatchValidation(MatchValidationTestBase):
    @classmethod
    def feature_group_class(cls) -> Any:
        return DateTimeFeatureGroup

    @classmethod
    def valid_operations(cls) -> set[str]:
        return set(DATETIME_OPS)

    @classmethod
    def config_key(cls) -> str:
        return "datetime_op"

    @classmethod
    def build_feature_name(cls, operation: str) -> str:
        return f"timestamp__{operation}"

    @classmethod
    def build_feature_name_no_source(cls) -> str:
        return "year"

    @classmethod
    def additional_match_options(cls) -> dict[str, Any]:
        return {"in_features": "timestamp"}
