"""Tests for WindowAggregationFeatureGroup base class."""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
    WindowAggregationFeatureGroup,
)


class TestClassAttributes:
    """Tests for WindowAggregationFeatureGroup class attributes."""

    def test_prefix_pattern_exists(self) -> None:
        """PREFIX_PATTERN regex attribute should be defined."""
        assert hasattr(WindowAggregationFeatureGroup, "PREFIX_PATTERN")
        assert isinstance(WindowAggregationFeatureGroup.PREFIX_PATTERN, str)

    def test_aggregation_types_exists(self) -> None:
        """AGGREGATION_TYPES dict should be defined with supported operations."""
        assert hasattr(WindowAggregationFeatureGroup, "AGGREGATION_TYPES")
        assert isinstance(WindowAggregationFeatureGroup.AGGREGATION_TYPES, dict)

    def test_aggregation_types_contains_standard_operations(self) -> None:
        """AGGREGATION_TYPES should contain standard aggregation operations."""
        expected_ops = {"sum", "avg", "count", "min", "max", "std", "var", "median"}
        for op in expected_ops:
            assert op in WindowAggregationFeatureGroup.AGGREGATION_TYPES, f"Missing standard operation: {op}"

    def test_aggregation_types_contains_advanced_operations(self) -> None:
        """AGGREGATION_TYPES should contain advanced aggregation operations."""
        expected_ops = {"mode", "nunique", "first", "last"}
        for op in expected_ops:
            assert op in WindowAggregationFeatureGroup.AGGREGATION_TYPES, f"Missing advanced operation: {op}"

    def test_min_in_features_is_one(self) -> None:
        """MIN_IN_FEATURES should be 1 (single source column)."""
        assert WindowAggregationFeatureGroup.MIN_IN_FEATURES == 1

    def test_max_in_features_is_one(self) -> None:
        """MAX_IN_FEATURES should be 1 (single source column)."""
        assert WindowAggregationFeatureGroup.MAX_IN_FEATURES == 1


class TestPatternMatching:
    """Tests for feature name pattern matching via match_feature_group_criteria."""

    @pytest.mark.parametrize(
        "feature_name",
        [
            "value_int__sum_groupby",
            "value_int__avg_groupby",
            "value_int__count_groupby",
            "value_int__min_groupby",
            "value_int__max_groupby",
            "value_int__std_groupby",
            "value_int__var_groupby",
            "value_int__median_groupby",
        ],
    )
    def test_matches_standard_operations(self, feature_name: str) -> None:
        """Standard aggregation operations with _groupby suffix should match."""
        options = Options(context={"partition_by": ["region"]})
        result = WindowAggregationFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is True, f"Should match: {feature_name}"

    @pytest.mark.parametrize(
        "feature_name",
        [
            "value_int__mode_groupby",
            "value_int__nunique_groupby",
            "value_int__first_groupby",
            "value_int__last_groupby",
        ],
    )
    def test_matches_advanced_operations(self, feature_name: str) -> None:
        """Advanced aggregation operations with _groupby suffix should match."""
        options = Options(context={"partition_by": ["region"]})
        result = WindowAggregationFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is True, f"Should match: {feature_name}"

    def test_no_match_wrong_suffix(self) -> None:
        """Feature with wrong suffix (grouped instead of groupby) should not match."""
        options = Options(context={"partition_by": ["region"]})
        result = WindowAggregationFeatureGroup.match_feature_group_criteria("value_int__avg_grouped", options, None)
        assert result is False

    def test_no_match_no_suffix(self) -> None:
        """Feature without _groupby suffix should not match."""
        options = Options(context={"partition_by": ["region"]})
        result = WindowAggregationFeatureGroup.match_feature_group_criteria("value_int__avg", options, None)
        assert result is False

    def test_no_match_no_source_column(self) -> None:
        """Feature with no source column (just operation_groupby) should not match."""
        options = Options(context={"partition_by": ["region"]})
        result = WindowAggregationFeatureGroup.match_feature_group_criteria("avg_groupby", options, None)
        assert result is False

    def test_no_match_invalid_operation(self) -> None:
        """Feature with an unknown/invalid operation should not match."""
        options = Options(context={"partition_by": ["region"]})
        result = WindowAggregationFeatureGroup.match_feature_group_criteria("value_int__unknown_groupby", options, None)
        assert result is False


class TestPatternParsing:
    """Tests for extracting operation and source column from feature names."""

    def test_parse_avg_operation(self) -> None:
        """Parsing value_int__avg_groupby should yield operation=avg, source=value_int."""
        operation = WindowAggregationFeatureGroup.get_aggregation_type("value_int__avg_groupby")
        assert operation == "avg"

    def test_parse_sum_operation(self) -> None:
        """Parsing my_col__sum_groupby should yield operation=sum, source=my_col."""
        operation = WindowAggregationFeatureGroup.get_aggregation_type("my_col__sum_groupby")
        assert operation == "sum"

    def test_parse_source_feature_from_avg(self) -> None:
        """Source feature should be extracted correctly from value_int__avg_groupby."""
        from mloda.user import Feature

        feature = Feature(
            "value_int__avg_groupby",
            options=Options(context={"partition_by": ["region"]}),
        )
        source_features = WindowAggregationFeatureGroup._extract_source_features(feature)
        assert source_features == ["value_int"]

    def test_parse_source_feature_from_sum(self) -> None:
        """Source feature should be extracted correctly from my_col__sum_groupby."""
        from mloda.user import Feature

        feature = Feature(
            "my_col__sum_groupby",
            options=Options(context={"partition_by": ["region"]}),
        )
        source_features = WindowAggregationFeatureGroup._extract_source_features(feature)
        assert source_features == ["my_col"]


class TestConfigValidation:
    """Tests for partition_by configuration validation."""

    def test_partition_by_required(self) -> None:
        """match_feature_group_criteria should fail without partition_by in options."""
        options = Options(context={})
        result = WindowAggregationFeatureGroup.match_feature_group_criteria("value_int__sum_groupby", options, None)
        assert result is False

    def test_partition_by_accepts_list_of_strings(self) -> None:
        """partition_by should accept a list of strings."""
        options = Options(context={"partition_by": ["region", "country"]})
        result = WindowAggregationFeatureGroup.match_feature_group_criteria("value_int__sum_groupby", options, None)
        assert result is True

    def test_partition_by_must_be_list(self) -> None:
        """partition_by as a plain string (not a list) should fail validation."""
        options = Options(context={"partition_by": "region"})
        result = WindowAggregationFeatureGroup.match_feature_group_criteria("value_int__sum_groupby", options, None)
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
        result = WindowAggregationFeatureGroup.match_feature_group_criteria("my_result", options, None)
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
        result = WindowAggregationFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False

    def test_config_based_match_rejects_missing_partition_by(self) -> None:
        """Config-based feature without partition_by should not match."""
        options = Options(
            context={
                "aggregation_type": "sum",
                "in_features": "value_int",
            }
        )
        result = WindowAggregationFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False

    def test_config_based_calculate_feature(self) -> None:
        """Config-based feature should compute correctly via calculate_feature."""
        import pyarrow as pa

        from mloda.core.abstract_plugins.components.feature_set import FeatureSet
        from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.pyarrow_window_aggregation import (
            PyArrowWindowAggregation,
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

        result = PyArrowWindowAggregation.calculate_feature(table, fs)
        assert isinstance(result, pa.Table)
        assert "my_sum_result" in result.column_names

        result_col = result.column("my_sum_result").to_pylist()
        expected = [25, 25, 25, 25, 140, 140, 140, 140, 70, 70, 70, -10]
        assert result_col == expected
