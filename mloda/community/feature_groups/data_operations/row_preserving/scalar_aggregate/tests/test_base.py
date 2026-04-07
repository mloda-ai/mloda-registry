"""Tests for ScalarAggregateFeatureGroup base class."""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.match_validation import MatchValidationTestBase
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.base import (
    AGGREGATION_TYPES,
    ScalarAggregateFeatureGroup,
)


class TestClassAttributes:
    def test_prefix_pattern_exists(self) -> None:
        assert hasattr(ScalarAggregateFeatureGroup, "PREFIX_PATTERN")
        assert isinstance(ScalarAggregateFeatureGroup.PREFIX_PATTERN, str)

    def test_aggregation_types_contains_all_operations(self) -> None:
        expected_ops = {"sum", "min", "max", "avg", "mean", "count", "std", "var", "median"}
        for op in expected_ops:
            assert op in AGGREGATION_TYPES, f"Missing operation: {op}"

    def test_min_in_features_is_one(self) -> None:
        assert ScalarAggregateFeatureGroup.MIN_IN_FEATURES == 1

    def test_max_in_features_is_one(self) -> None:
        assert ScalarAggregateFeatureGroup.MAX_IN_FEATURES == 1


class TestPatternMatching:
    @pytest.mark.parametrize(
        "feature_name",
        [
            "value_int__sum_scalar",
            "value_int__min_scalar",
            "value_int__max_scalar",
            "value_int__avg_scalar",
            "value_int__mean_scalar",
            "value_int__count_scalar",
            "value_int__std_scalar",
            "value_int__var_scalar",
            "value_int__median_scalar",
        ],
    )
    def test_matches_all_operations(self, feature_name: str) -> None:
        options = Options()
        result = ScalarAggregateFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is True, f"Should match: {feature_name}"

    def test_no_match_wrong_suffix(self) -> None:
        options = Options()
        result = ScalarAggregateFeatureGroup.match_feature_group_criteria("value_int__sum_grouped", options, None)
        assert result is False

    def test_no_match_no_suffix(self) -> None:
        options = Options()
        result = ScalarAggregateFeatureGroup.match_feature_group_criteria("value_int__sum", options, None)
        assert result is False

    def test_no_match_no_source_column(self) -> None:
        options = Options()
        result = ScalarAggregateFeatureGroup.match_feature_group_criteria("sum_scalar", options, None)
        assert result is False

    def test_no_match_invalid_operation(self) -> None:
        options = Options()
        result = ScalarAggregateFeatureGroup.match_feature_group_criteria("value_int__unknown_scalar", options, None)
        assert result is False


class TestPatternParsing:
    def test_parse_sum_operation(self) -> None:
        operation = ScalarAggregateFeatureGroup.get_aggregation_type("value_int__sum_scalar")
        assert operation == "sum"

    def test_parse_avg_operation(self) -> None:
        operation = ScalarAggregateFeatureGroup.get_aggregation_type("value_int__avg_scalar")
        assert operation == "avg"

    def test_parse_source_feature(self) -> None:
        from mloda.user import Feature

        feature = Feature("value_int__sum_scalar", options=Options())
        source_features = ScalarAggregateFeatureGroup._extract_source_features(feature)
        assert source_features == ["value_int"]

    def test_parse_source_feature_with_underscores(self) -> None:
        from mloda.user import Feature

        feature = Feature("my_value__max_scalar", options=Options())
        source_features = ScalarAggregateFeatureGroup._extract_source_features(feature)
        assert source_features == ["my_value"]


class TestConfigBasedFeatures:
    def test_config_based_match(self) -> None:
        options = Options(
            context={
                "aggregation_type": "sum",
                "in_features": "value_int",
            }
        )
        result = ScalarAggregateFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is True

    def test_config_based_match_rejects_invalid_op(self) -> None:
        options = Options(
            context={
                "aggregation_type": "invalid_op",
                "in_features": "value_int",
            }
        )
        result = ScalarAggregateFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False


class TestSingleColumnEnforcement:
    """Verify that MAX_IN_FEATURES=1 enforces single-column behavior.

    The aggregation package computes a scalar aggregate over one source
    column and broadcasts it to every row. Multiple in_features are
    rejected at two levels: input_features() validates the count during
    feature resolution, and _extract_source_features() validates it
    again during calculate_feature() to prevent silent truncation.
    """

    def test_max_in_features_is_one(self) -> None:
        assert ScalarAggregateFeatureGroup.MAX_IN_FEATURES == 1

    def test_input_features_rejects_multiple_option_in_features(self) -> None:
        """Option-based features with >1 in_features must be rejected."""
        options = Options(
            context={
                "aggregation_type": "sum",
                "in_features": ["col_a", "col_b"],
            }
        )
        instance = ScalarAggregateFeatureGroup()
        with pytest.raises(ValueError, match="at most 1"):
            instance.input_features(options, FeatureName("my_result"))

    def test_extract_source_features_rejects_multiple_in_features(self) -> None:
        """_extract_source_features must reject multiple source features.

        This guards against silent truncation to a single column if
        calculate_feature() is invoked with multi-column options that
        bypassed input_features() validation.
        """
        options = Options(
            context={
                "aggregation_type": "sum",
                "in_features": ["col_a", "col_b"],
            }
        )
        feature = Feature("my_result", options=options)
        with pytest.raises(ValueError, match="at most 1"):
            ScalarAggregateFeatureGroup._extract_source_features(feature)

    def test_extract_source_features_returns_single_item_for_string_pattern(self) -> None:
        feature = Feature("value_int__sum_scalar", options=Options())
        source_features = ScalarAggregateFeatureGroup._extract_source_features(feature)
        assert len(source_features) == 1
        assert source_features == ["value_int"]

    def test_extract_source_features_returns_single_item_for_option_config(self) -> None:
        """Option-based config with one in_feature returns a single-element list."""
        options = Options(
            context={
                "aggregation_type": "max",
                "in_features": "revenue",
            }
        )
        feature = Feature("my_result", options=options)
        source_features = ScalarAggregateFeatureGroup._extract_source_features(feature)
        assert len(source_features) == 1
        assert source_features == ["revenue"]

    def test_input_features_returns_single_feature_for_string_pattern(self) -> None:
        options = Options()
        instance = ScalarAggregateFeatureGroup()
        result = instance.input_features(options, FeatureName("value_int__sum_scalar"))
        assert result is not None
        assert len(result) == 1
        names = {f.name for f in result}
        assert names == {"value_int"}

    def test_input_features_returns_single_feature_for_option_config(self) -> None:
        options = Options(
            context={
                "aggregation_type": "max",
                "in_features": "revenue",
            }
        )
        instance = ScalarAggregateFeatureGroup()
        result = instance.input_features(options, FeatureName("my_max_result"))
        assert result is not None
        assert len(result) == 1
        names = {f.name for f in result}
        assert names == {"revenue"}


class TestAggregationTypeExtraction:
    """Verify aggregation type extraction from both string and option sources."""

    def test_get_aggregation_type_raises_for_non_pattern_name(self) -> None:
        with pytest.raises(ValueError, match="Could not extract"):
            ScalarAggregateFeatureGroup.get_aggregation_type("plain_name")

    def test_extract_aggregation_type_from_options(self) -> None:
        options = Options(
            context={
                "aggregation_type": "median",
                "in_features": "value_int",
            }
        )
        feature = Feature("my_result", options=options)
        agg_type = ScalarAggregateFeatureGroup._extract_aggregation_type(feature)
        assert agg_type == "median"

    def test_extract_aggregation_type_raises_without_option(self) -> None:
        feature = Feature("plain_name", options=Options())
        with pytest.raises(ValueError, match="Could not extract"):
            ScalarAggregateFeatureGroup._extract_aggregation_type(feature)

    @pytest.mark.parametrize("agg_type", list(AGGREGATION_TYPES.keys()))
    def test_get_aggregation_type_for_all_ops(self, agg_type: str) -> None:
        feature_name = f"col__{agg_type}_scalar"
        result = ScalarAggregateFeatureGroup.get_aggregation_type(feature_name)
        assert result == agg_type


class TestScalarAggregateMatchValidation(MatchValidationTestBase):
    """Shared match-validation tests adapted for scalar aggregate."""

    @classmethod
    def feature_group_class(cls) -> Any:
        return ScalarAggregateFeatureGroup

    @classmethod
    def valid_operations(cls) -> set[str]:
        return set(AGGREGATION_TYPES)

    @classmethod
    def config_key(cls) -> str:
        return "aggregation_type"

    @classmethod
    def build_feature_name(cls, operation: str) -> str:
        return f"value_int__{operation}_scalar"

    @classmethod
    def build_feature_name_no_source(cls) -> str:
        return "sum_scalar"

    @classmethod
    def additional_match_options(cls) -> dict[str, Any]:
        return {"in_features": "value_int"}
