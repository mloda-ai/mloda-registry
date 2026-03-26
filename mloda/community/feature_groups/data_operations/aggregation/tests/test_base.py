"""Tests for ColumnAggregationFeatureGroup base class."""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.components.options import Options

from mloda.community.feature_groups.data_operations.aggregation.base import (
    AGGREGATION_TYPES,
    ColumnAggregationFeatureGroup,
)


class TestClassAttributes:
    def test_prefix_pattern_exists(self) -> None:
        assert hasattr(ColumnAggregationFeatureGroup, "PREFIX_PATTERN")
        assert isinstance(ColumnAggregationFeatureGroup.PREFIX_PATTERN, str)

    def test_aggregation_types_contains_all_operations(self) -> None:
        expected_ops = {"sum", "min", "max", "avg", "mean", "count", "std", "var", "median"}
        for op in expected_ops:
            assert op in AGGREGATION_TYPES, f"Missing operation: {op}"

    def test_min_in_features_is_one(self) -> None:
        assert ColumnAggregationFeatureGroup.MIN_IN_FEATURES == 1

    def test_max_in_features_is_one(self) -> None:
        assert ColumnAggregationFeatureGroup.MAX_IN_FEATURES == 1


class TestPatternMatching:
    @pytest.mark.parametrize(
        "feature_name",
        [
            "value_int__sum_aggr",
            "value_int__min_aggr",
            "value_int__max_aggr",
            "value_int__avg_aggr",
            "value_int__mean_aggr",
            "value_int__count_aggr",
            "value_int__std_aggr",
            "value_int__var_aggr",
            "value_int__median_aggr",
        ],
    )
    def test_matches_all_operations(self, feature_name: str) -> None:
        options = Options()
        result = ColumnAggregationFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is True, f"Should match: {feature_name}"

    def test_no_match_wrong_suffix(self) -> None:
        options = Options()
        result = ColumnAggregationFeatureGroup.match_feature_group_criteria("value_int__sum_grouped", options, None)
        assert result is False

    def test_no_match_no_suffix(self) -> None:
        options = Options()
        result = ColumnAggregationFeatureGroup.match_feature_group_criteria("value_int__sum", options, None)
        assert result is False

    def test_no_match_no_source_column(self) -> None:
        options = Options()
        result = ColumnAggregationFeatureGroup.match_feature_group_criteria("sum_aggr", options, None)
        assert result is False

    def test_no_match_invalid_operation(self) -> None:
        options = Options()
        result = ColumnAggregationFeatureGroup.match_feature_group_criteria("value_int__unknown_aggr", options, None)
        assert result is False


class TestPatternParsing:
    def test_parse_sum_operation(self) -> None:
        operation = ColumnAggregationFeatureGroup.get_aggregation_type("value_int__sum_aggr")
        assert operation == "sum"

    def test_parse_avg_operation(self) -> None:
        operation = ColumnAggregationFeatureGroup.get_aggregation_type("value_int__avg_aggr")
        assert operation == "avg"

    def test_parse_source_feature(self) -> None:
        from mloda.user import Feature

        feature = Feature("value_int__sum_aggr", options=Options())
        source_features = ColumnAggregationFeatureGroup._extract_source_features(feature)
        assert source_features == ["value_int"]

    def test_parse_source_feature_with_underscores(self) -> None:
        from mloda.user import Feature

        feature = Feature("my_value__max_aggr", options=Options())
        source_features = ColumnAggregationFeatureGroup._extract_source_features(feature)
        assert source_features == ["my_value"]


class TestConfigBasedFeatures:
    def test_config_based_match(self) -> None:
        options = Options(
            context={
                "aggregation_type": "sum",
                "in_features": "value_int",
            }
        )
        result = ColumnAggregationFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is True

    def test_config_based_match_rejects_invalid_op(self) -> None:
        options = Options(
            context={
                "aggregation_type": "invalid_op",
                "in_features": "value_int",
            }
        )
        result = ColumnAggregationFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False
