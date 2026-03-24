"""Tests for GroupAggregationFeatureGroup base class."""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.community.feature_groups.data_operations.group_aggregation.base import (
    GroupAggregationFeatureGroup,
)


class TestClassAttributes:
    """Tests for GroupAggregationFeatureGroup class attributes."""

    def test_prefix_pattern_exists(self) -> None:
        """PREFIX_PATTERN regex attribute should be defined."""
        assert hasattr(GroupAggregationFeatureGroup, "PREFIX_PATTERN")
        assert isinstance(GroupAggregationFeatureGroup.PREFIX_PATTERN, str)

    def test_aggregation_types_exists(self) -> None:
        """AGGREGATION_TYPES dict should be defined with supported operations."""
        assert hasattr(GroupAggregationFeatureGroup, "AGGREGATION_TYPES")
        assert isinstance(GroupAggregationFeatureGroup.AGGREGATION_TYPES, dict)

    def test_aggregation_types_contains_standard_operations(self) -> None:
        """AGGREGATION_TYPES should contain standard aggregation operations."""
        expected_ops = {"sum", "avg", "count", "min", "max", "std", "var", "median"}
        for op in expected_ops:
            assert op in GroupAggregationFeatureGroup.AGGREGATION_TYPES, f"Missing standard operation: {op}"

    def test_aggregation_types_contains_advanced_operations(self) -> None:
        """AGGREGATION_TYPES should contain advanced aggregation operations."""
        expected_ops = {"mode", "nunique", "first", "last"}
        for op in expected_ops:
            assert op in GroupAggregationFeatureGroup.AGGREGATION_TYPES, f"Missing advanced operation: {op}"

    def test_min_in_features_is_one(self) -> None:
        """MIN_IN_FEATURES should be 1 (single source column)."""
        assert GroupAggregationFeatureGroup.MIN_IN_FEATURES == 1

    def test_max_in_features_is_one(self) -> None:
        """MAX_IN_FEATURES should be 1 (single source column)."""
        assert GroupAggregationFeatureGroup.MAX_IN_FEATURES == 1


class TestPatternMatching:
    """Tests for feature name pattern matching via match_feature_group_criteria."""

    @pytest.mark.parametrize(
        "feature_name",
        [
            "value_int__sum_grouped",
            "value_int__avg_grouped",
            "value_int__count_grouped",
            "value_int__min_grouped",
            "value_int__max_grouped",
            "value_int__std_grouped",
            "value_int__var_grouped",
            "value_int__median_grouped",
        ],
    )
    def test_matches_standard_operations(self, feature_name: str) -> None:
        """Standard aggregation operations with _grouped suffix should match."""
        options = Options(context={"partition_by": ["region"]})
        result = GroupAggregationFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is True, f"Should match: {feature_name}"

    @pytest.mark.parametrize(
        "feature_name",
        [
            "value_int__mode_grouped",
            "value_int__nunique_grouped",
            "value_int__first_grouped",
            "value_int__last_grouped",
        ],
    )
    def test_matches_advanced_operations(self, feature_name: str) -> None:
        """Advanced aggregation operations with _grouped suffix should match."""
        options = Options(context={"partition_by": ["region"]})
        result = GroupAggregationFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is True, f"Should match: {feature_name}"

    def test_rejects_unimplemented_dynamic_type(self) -> None:
        """Unimplemented dynamic types like percentile_75 should not match."""
        options = Options(context={"partition_by": ["region"]})
        result = GroupAggregationFeatureGroup.match_feature_group_criteria(
            "value_int__percentile_75_grouped", options, None
        )
        assert result is False

    def test_no_match_wrong_suffix(self) -> None:
        """Feature with wrong suffix (groupby instead of grouped) should not match."""
        options = Options(context={"partition_by": ["region"]})
        result = GroupAggregationFeatureGroup.match_feature_group_criteria("value_int__avg_groupby", options, None)
        assert result is False

    def test_no_match_no_suffix(self) -> None:
        """Feature without _grouped suffix should not match."""
        options = Options(context={"partition_by": ["region"]})
        result = GroupAggregationFeatureGroup.match_feature_group_criteria("value_int__avg", options, None)
        assert result is False

    def test_no_match_no_source_column(self) -> None:
        """Feature with no source column (just operation_grouped) should not match."""
        options = Options(context={"partition_by": ["region"]})
        result = GroupAggregationFeatureGroup.match_feature_group_criteria("avg_grouped", options, None)
        assert result is False

    def test_no_match_invalid_operation(self) -> None:
        """Feature with an unknown/invalid operation should not match."""
        options = Options(context={"partition_by": ["region"]})
        result = GroupAggregationFeatureGroup.match_feature_group_criteria("value_int__unknown_grouped", options, None)
        assert result is False


class TestPatternParsing:
    """Tests for extracting operation and source column from feature names."""

    def test_parse_avg_operation(self) -> None:
        """Parsing value_int__avg_grouped should yield operation=avg."""
        operation = GroupAggregationFeatureGroup.get_aggregation_type("value_int__avg_grouped")
        assert operation == "avg"

    def test_parse_sum_operation(self) -> None:
        """Parsing my_col__sum_grouped should yield operation=sum."""
        operation = GroupAggregationFeatureGroup.get_aggregation_type("my_col__sum_grouped")
        assert operation == "sum"

    def test_parse_percentile_operation(self) -> None:
        """Parsing value_int__percentile_75_grouped should yield operation=percentile_75."""
        operation = GroupAggregationFeatureGroup.get_aggregation_type("value_int__percentile_75_grouped")
        assert operation == "percentile_75"

    def test_parse_source_feature_from_avg(self) -> None:
        """Source feature should be extracted correctly from value_int__avg_grouped."""
        from mloda.user import Feature

        feature = Feature(
            "value_int__avg_grouped",
            options=Options(context={"partition_by": ["region"]}),
        )
        source_features = GroupAggregationFeatureGroup._extract_source_features(feature)
        assert source_features == ["value_int"]

    def test_parse_source_feature_from_sum(self) -> None:
        """Source feature should be extracted correctly from my_col__sum_grouped."""
        from mloda.user import Feature

        feature = Feature(
            "my_col__sum_grouped",
            options=Options(context={"partition_by": ["region"]}),
        )
        source_features = GroupAggregationFeatureGroup._extract_source_features(feature)
        assert source_features == ["my_col"]


class TestConfigValidation:
    """Tests for partition_by configuration validation."""

    def test_partition_by_required(self) -> None:
        """match_feature_group_criteria should fail without partition_by in options."""
        options = Options(context={})
        result = GroupAggregationFeatureGroup.match_feature_group_criteria("value_int__sum_grouped", options, None)
        assert result is False

    def test_partition_by_accepts_list_of_strings(self) -> None:
        """partition_by should accept a list of strings."""
        options = Options(context={"partition_by": ["region", "country"]})
        result = GroupAggregationFeatureGroup.match_feature_group_criteria("value_int__sum_grouped", options, None)
        assert result is True

    def test_partition_by_must_be_list(self) -> None:
        """partition_by as a plain string (not a list) should fail validation."""
        options = Options(context={"partition_by": "region"})
        result = GroupAggregationFeatureGroup.match_feature_group_criteria("value_int__sum_grouped", options, None)
        assert result is False


class TestConfigBasedFeatures:
    """Tests for configuration-based feature matching (non-string features)."""

    def test_config_based_match(self) -> None:
        """A feature with aggregation_type and in_features in options should match."""
        options = Options(
            context={
                "aggregation_type": "sum",
                "in_features": "value_int",
                "partition_by": ["region"],
            }
        )
        result = GroupAggregationFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is True

    def test_config_based_match_rejects_multiple_in_features(self) -> None:
        """Config-based feature with multiple in_features should not match (MAX_IN_FEATURES=1)."""
        from mloda.user import Feature as UserFeature

        options = Options(
            context={
                "aggregation_type": "sum",
                "in_features": frozenset({UserFeature("value_int"), UserFeature("value_float")}),
                "partition_by": ["region"],
            }
        )
        result = GroupAggregationFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False

    def test_config_based_match_rejects_missing_partition_by(self) -> None:
        """Config-based feature without partition_by should not match."""
        options = Options(
            context={
                "aggregation_type": "sum",
                "in_features": "value_int",
            }
        )
        result = GroupAggregationFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False

    def test_config_based_calculate_feature(self) -> None:
        """Config-based feature should compute correctly via calculate_feature."""
        import pyarrow as pa

        from mloda.core.abstract_plugins.components.feature_set import FeatureSet
        from mloda.community.feature_groups.data_operations.group_aggregation.pyarrow_group_aggregation import (
            PyArrowGroupAggregation,
        )
        from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
        from mloda.user import Feature

        table = PyArrowDataOpsTestDataCreator.create()

        feature = Feature(
            "my_sum_result",
            options=Options(
                context={
                    "aggregation_type": "sum",
                    "in_features": "value_int",
                    "partition_by": ["region"],
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = PyArrowGroupAggregation.calculate_feature(table, fs)
        assert isinstance(result, pa.Table)
        assert "my_sum_result" in result.column_names
        assert result.num_rows == 4

        region_col = result.column("region").to_pylist()
        result_col = result.column("my_sum_result").to_pylist()
        result_map = {region_col[i]: result_col[i] for i in range(len(region_col))}
        assert result_map["A"] == 25
        assert result_map["B"] == 140
        assert result_map["C"] == 70
        assert result_map[None] == -10
