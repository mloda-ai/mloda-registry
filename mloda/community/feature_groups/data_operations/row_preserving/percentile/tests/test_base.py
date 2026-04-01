"""Tests for PercentileFeatureGroup base class."""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.match_validation import MatchValidationTestBase
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.row_preserving.percentile.base import (
    PercentileFeatureGroup,
)


class TestClassAttributes:
    def test_prefix_pattern_exists(self) -> None:
        assert hasattr(PercentileFeatureGroup, "PREFIX_PATTERN")
        assert isinstance(PercentileFeatureGroup.PREFIX_PATTERN, str)

    def test_min_in_features_is_one(self) -> None:
        assert PercentileFeatureGroup.MIN_IN_FEATURES == 1

    def test_max_in_features_is_one(self) -> None:
        assert PercentileFeatureGroup.MAX_IN_FEATURES == 1

    def test_percentile_constant_defined(self) -> None:
        assert PercentileFeatureGroup.PERCENTILE == "percentile"

    def test_partition_by_constant_defined(self) -> None:
        assert PercentileFeatureGroup.PARTITION_BY == "partition_by"


class TestPatternMatching:
    @pytest.mark.parametrize(
        "feature_name",
        [
            "value_int__p0_percentile",
            "value_int__p25_percentile",
            "value_int__p50_percentile",
            "value_int__p75_percentile",
            "value_int__p90_percentile",
            "value_int__p95_percentile",
            "value_int__p99_percentile",
            "value_int__p100_percentile",
        ],
    )
    def test_matches_valid_percentiles(self, feature_name: str) -> None:
        options = Options(context={"partition_by": ["region"]})
        result = PercentileFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is True, f"Should match: {feature_name}"

    def test_no_match_wrong_suffix(self) -> None:
        options = Options(context={"partition_by": ["region"]})
        result = PercentileFeatureGroup.match_feature_group_criteria("value_int__p50_grouped", options, None)
        assert result is False

    def test_no_match_no_suffix(self) -> None:
        options = Options(context={"partition_by": ["region"]})
        result = PercentileFeatureGroup.match_feature_group_criteria("value_int__p50", options, None)
        assert result is False

    def test_no_match_no_source_column(self) -> None:
        options = Options(context={"partition_by": ["region"]})
        result = PercentileFeatureGroup.match_feature_group_criteria("p50_percentile", options, None)
        assert result is False

    def test_no_match_invalid_percentile_too_high(self) -> None:
        options = Options(context={"partition_by": ["region"]})
        result = PercentileFeatureGroup.match_feature_group_criteria("value_int__p101_percentile", options, None)
        assert result is False

    def test_no_match_invalid_percentile_negative(self) -> None:
        options = Options(context={"partition_by": ["region"]})
        result = PercentileFeatureGroup.match_feature_group_criteria("value_int__p-1_percentile", options, None)
        assert result is False

    def test_no_match_non_numeric(self) -> None:
        options = Options(context={"partition_by": ["region"]})
        result = PercentileFeatureGroup.match_feature_group_criteria("value_int__pfoo_percentile", options, None)
        assert result is False


class TestPatternParsing:
    def test_parse_p50(self) -> None:
        result = PercentileFeatureGroup.get_percentile_value("value_int__p50_percentile")
        assert result == 0.5

    def test_parse_p25(self) -> None:
        result = PercentileFeatureGroup.get_percentile_value("value_int__p25_percentile")
        assert result == 0.25

    def test_parse_p0(self) -> None:
        result = PercentileFeatureGroup.get_percentile_value("value_int__p0_percentile")
        assert result == 0.0

    def test_parse_p100(self) -> None:
        result = PercentileFeatureGroup.get_percentile_value("value_int__p100_percentile")
        assert result == 1.0

    def test_parse_source_feature(self) -> None:
        feature = Feature(
            "value_int__p50_percentile",
            options=Options(context={"partition_by": ["region"]}),
        )
        source_features = PercentileFeatureGroup._extract_source_features(feature)
        assert source_features == ["value_int"]

    def test_parse_source_feature_with_underscores(self) -> None:
        feature = Feature(
            "my_value__p75_percentile",
            options=Options(context={"partition_by": ["region"]}),
        )
        source_features = PercentileFeatureGroup._extract_source_features(feature)
        assert source_features == ["my_value"]


class TestConfigBasedFeatures:
    def test_config_based_match(self) -> None:
        options = Options(
            context={
                "percentile": 0.75,
                "in_features": "value_int",
                "partition_by": ["region"],
            }
        )
        result = PercentileFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is True

    def test_config_based_match_rejects_invalid_percentile_too_high(self) -> None:
        options = Options(
            context={
                "percentile": 1.5,
                "in_features": "value_int",
                "partition_by": ["region"],
            }
        )
        result = PercentileFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False

    def test_config_based_match_rejects_invalid_percentile_negative(self) -> None:
        options = Options(
            context={
                "percentile": -0.1,
                "in_features": "value_int",
                "partition_by": ["region"],
            }
        )
        result = PercentileFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False

    def test_config_based_match_rejects_missing_partition_by(self) -> None:
        options = Options(
            context={
                "percentile": 0.5,
                "in_features": "value_int",
            }
        )
        result = PercentileFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False


class TestPercentileMatchValidation(MatchValidationTestBase):
    @classmethod
    def feature_group_class(cls) -> Any:
        return PercentileFeatureGroup

    @classmethod
    def valid_operations(cls) -> set[str]:
        return {"p0", "p25", "p50", "p75", "p90", "p95", "p99", "p100"}

    @classmethod
    def config_key(cls) -> str:
        return "percentile"

    @classmethod
    def build_feature_name(cls, operation: str) -> str:
        return f"value_int__{operation}_percentile"

    @classmethod
    def build_feature_name_no_source(cls) -> str:
        return "p50_percentile"

    @classmethod
    def additional_match_options(cls) -> dict[str, Any]:
        return {"in_features": "value_int", "partition_by": ["region"]}

    @classmethod
    def pattern_match_options(cls) -> Options:
        return Options(context={"partition_by": ["region"]})

    @classmethod
    def options_reject_invalid_types(cls) -> bool:
        return False
