"""Tests for PercentileFeatureGroup base class."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.match_validation import MatchValidationTestBase, TokenCase
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

    def test_config_based_match_rejects_bool_true(self) -> None:
        options = Options(
            context={
                "percentile": True,
                "in_features": "value_int",
                "partition_by": ["region"],
            }
        )
        result = PercentileFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False

    def test_config_based_match_rejects_bool_false(self) -> None:
        options = Options(
            context={
                "percentile": False,
                "in_features": "value_int",
                "partition_by": ["region"],
            }
        )
        result = PercentileFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False

    def test_config_based_match_boundary_int_one(self) -> None:
        options = Options(
            context={
                "percentile": 1,
                "in_features": "value_int",
                "partition_by": ["region"],
            }
        )
        result = PercentileFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is True

    def test_config_based_match_boundary_int_zero(self) -> None:
        options = Options(
            context={
                "percentile": 0,
                "in_features": "value_int",
                "partition_by": ["region"],
            }
        )
        result = PercentileFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is True

    def test_config_based_match_boundary_float_one(self) -> None:
        options = Options(
            context={
                "percentile": 1.0,
                "in_features": "value_int",
                "partition_by": ["region"],
            }
        )
        result = PercentileFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is True

    def test_config_based_match_boundary_float_zero(self) -> None:
        options = Options(
            context={
                "percentile": 0.0,
                "in_features": "value_int",
                "partition_by": ["region"],
            }
        )
        result = PercentileFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is True

    def test_config_based_match_rejects_missing_partition_by(self) -> None:
        options = Options(
            context={
                "percentile": 0.5,
                "in_features": "value_int",
            }
        )
        result = PercentileFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False


class TestConfigBasedExtraction:
    def test_extract_rejects_bool_true(self) -> None:
        feature = Feature(
            "my_result",
            options=Options(context={"percentile": True, "partition_by": ["region"]}),
        )
        with pytest.raises(ValueError):
            PercentileFeatureGroup._extract_percentile(feature)

    def test_extract_rejects_bool_false(self) -> None:
        feature = Feature(
            "my_result",
            options=Options(context={"percentile": False, "partition_by": ["region"]}),
        )
        with pytest.raises(ValueError):
            PercentileFeatureGroup._extract_percentile(feature)

    def test_extract_float_value(self) -> None:
        feature = Feature(
            "my_result",
            options=Options(context={"percentile": 0.75, "partition_by": ["region"]}),
        )
        result = PercentileFeatureGroup._extract_percentile(feature)
        assert result == 0.75

    def test_extract_int_value(self) -> None:
        feature = Feature(
            "my_result",
            options=Options(context={"percentile": 1, "partition_by": ["region"]}),
        )
        result = PercentileFeatureGroup._extract_percentile(feature)
        assert result == 1.0


class TestRejectionReporting:
    """A wrong-typed percentile is a reported strict-validation rejection, not a silent non-match."""

    def test_wrong_typed_percentile_reports_a_reason(self) -> None:
        options = Options(
            context={
                "percentile": "fifty",
                "in_features": "value_int",
                "partition_by": ["region"],
            }
        )
        reason = PercentileFeatureGroup._strict_validation_rejection_reason("my_result", options)
        assert reason is not None
        assert "percentile" in reason

    def test_out_of_range_percentile_reports_a_reason(self) -> None:
        """An out-of-range percentile is a reported strict-validation rejection, not a silent non-match."""
        for value in (50, 1.5):
            options = Options(
                context={
                    "percentile": value,
                    "in_features": "value_int",
                    "partition_by": ["region"],
                }
            )
            assert PercentileFeatureGroup.match_feature_group_criteria("my_result", options, None) is False
            reason = PercentileFeatureGroup._strict_validation_rejection_reason("my_result", options)
            assert reason is not None
            assert "percentile" in reason

    def test_valid_percentile_reports_nothing(self) -> None:
        options = Options(
            context={
                "percentile": 0.5,
                "in_features": "value_int",
                "partition_by": ["region"],
            }
        )
        reason = PercentileFeatureGroup._strict_validation_rejection_reason("my_result", options)
        assert reason is None


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
    def config_value(cls, operation: str) -> Any:
        # The name path uses pN tokens, the config path a float in [0.0, 1.0].
        return int(operation[1:]) / 100.0

    @classmethod
    def options_reject_invalid_types(cls) -> bool:
        # percentile is strict with an element validator, so wrong-typed config values are rejected.
        return True

    @classmethod
    def token_cases(cls) -> list[TokenCase]:
        # A percentile is one float in [0.0, 1.0]: out of range, a bool, and an int too large
        # for float all stay out. The last one must be a plain non-match rather than an
        # OverflowError, which the mixin does not catch and which would abort discovery for
        # every feature group.
        primary = super().token_cases()[0]
        return [replace(primary, invalid=(1.5, True, 10**400))]

    @classmethod
    def dispatch_values(cls, options: Options) -> list[Any]:
        return [PercentileFeatureGroup._extract_percentile(Feature("my_result", options=options))]
