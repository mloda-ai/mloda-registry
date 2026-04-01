"""Tests for FilteredAggregationFeatureGroup base class."""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.match_validation import MatchValidationTestBase

from mloda.community.feature_groups.data_operations.row_preserving.filtered_aggregation.base import (
    FilteredAggregationFeatureGroup,
)

_VALID_OPTIONS = Options(
    context={
        "partition_by": ["region"],
        "filter_column": "category",
        "filter_value": "X",
    }
)


class TestClassAttributes:
    """Tests for FilteredAggregationFeatureGroup class attributes."""

    def test_prefix_pattern_exists(self) -> None:
        """PREFIX_PATTERN regex attribute should be defined."""
        assert hasattr(FilteredAggregationFeatureGroup, "PREFIX_PATTERN")
        assert isinstance(FilteredAggregationFeatureGroup.PREFIX_PATTERN, str)

    def test_aggregation_types_exists(self) -> None:
        """AGGREGATION_TYPES dict should be defined with supported operations."""
        assert hasattr(FilteredAggregationFeatureGroup, "AGGREGATION_TYPES")
        assert isinstance(FilteredAggregationFeatureGroup.AGGREGATION_TYPES, dict)

    def test_aggregation_types_contains_standard_operations(self) -> None:
        """AGGREGATION_TYPES should contain standard aggregation operations."""
        expected_ops = {"sum", "avg", "count", "min", "max"}
        for op in expected_ops:
            assert op in FilteredAggregationFeatureGroup.AGGREGATION_TYPES, f"Missing operation: {op}"

    def test_aggregation_types_contains_mean_alias(self) -> None:
        """AGGREGATION_TYPES should contain mean as an alias for avg."""
        assert "mean" in FilteredAggregationFeatureGroup.AGGREGATION_TYPES

    def test_min_in_features_is_one(self) -> None:
        """MIN_IN_FEATURES should be 1 (single source column)."""
        assert FilteredAggregationFeatureGroup.MIN_IN_FEATURES == 1

    def test_max_in_features_is_one(self) -> None:
        """MAX_IN_FEATURES should be 1 (single source column)."""
        assert FilteredAggregationFeatureGroup.MAX_IN_FEATURES == 1


class TestPatternMatching:
    """Tests for feature name pattern matching via match_feature_group_criteria."""

    @pytest.mark.parametrize(
        "feature_name",
        [
            "value_int__sum_filtered_groupby",
            "value_int__avg_filtered_groupby",
            "value_int__count_filtered_groupby",
            "value_int__min_filtered_groupby",
            "value_int__max_filtered_groupby",
            "value_int__mean_filtered_groupby",
        ],
    )
    def test_matches_standard_operations(self, feature_name: str) -> None:
        """Standard aggregation operations with _filtered_groupby suffix should match."""
        result = FilteredAggregationFeatureGroup.match_feature_group_criteria(feature_name, _VALID_OPTIONS, None)
        assert result is True, f"Should match: {feature_name}"

    def test_no_match_wrong_suffix(self) -> None:
        """Feature with wrong suffix should not match."""
        result = FilteredAggregationFeatureGroup.match_feature_group_criteria(
            "value_int__avg_window", _VALID_OPTIONS, None
        )
        assert result is False

    def test_no_match_no_suffix(self) -> None:
        """Feature without _filtered_groupby suffix should not match."""
        result = FilteredAggregationFeatureGroup.match_feature_group_criteria("value_int__avg", _VALID_OPTIONS, None)
        assert result is False

    def test_no_match_no_source_column(self) -> None:
        """Feature with no source column (just operation_filtered_groupby) should not match."""
        result = FilteredAggregationFeatureGroup.match_feature_group_criteria(
            "avg_filtered_groupby", _VALID_OPTIONS, None
        )
        assert result is False

    def test_no_match_invalid_operation(self) -> None:
        """Feature with an unknown/invalid operation should not match."""
        result = FilteredAggregationFeatureGroup.match_feature_group_criteria(
            "value_int__unknown_filtered_groupby", _VALID_OPTIONS, None
        )
        assert result is False


class TestPatternParsing:
    """Tests for extracting operation and source column from feature names."""

    def test_parse_sum_operation(self) -> None:
        """Parsing value_int__sum_filtered_groupby should yield operation=sum."""
        operation = FilteredAggregationFeatureGroup.get_aggregation_type("value_int__sum_filtered_groupby")
        assert operation == "sum"

    def test_parse_avg_operation(self) -> None:
        """Parsing my_col__avg_filtered_groupby should yield operation=avg."""
        operation = FilteredAggregationFeatureGroup.get_aggregation_type("my_col__avg_filtered_groupby")
        assert operation == "avg"

    def test_parse_source_feature_from_sum(self) -> None:
        """Source feature should be extracted correctly from value_int__sum_filtered_groupby."""
        from mloda.user import Feature

        feature = Feature("value_int__sum_filtered_groupby", options=_VALID_OPTIONS)
        source_features = FilteredAggregationFeatureGroup._extract_source_features(feature)
        assert source_features == ["value_int"]

    def test_parse_source_feature_from_avg(self) -> None:
        """Source feature should be extracted correctly from my_col__avg_filtered_groupby."""
        from mloda.user import Feature

        feature = Feature("my_col__avg_filtered_groupby", options=_VALID_OPTIONS)
        source_features = FilteredAggregationFeatureGroup._extract_source_features(feature)
        assert source_features == ["my_col"]


class TestConfigValidation:
    """Tests for partition_by, filter_column, and filter_value configuration validation."""

    def test_partition_by_required(self) -> None:
        """match_feature_group_criteria should fail without partition_by in options."""
        options = Options(context={"filter_column": "category", "filter_value": "X"})
        result = FilteredAggregationFeatureGroup.match_feature_group_criteria(
            "value_int__sum_filtered_groupby", options, None
        )
        assert result is False

    def test_partition_by_accepts_list_of_strings(self) -> None:
        """partition_by should accept a list of strings."""
        options = Options(
            context={
                "partition_by": ["region", "country"],
                "filter_column": "category",
                "filter_value": "X",
            }
        )
        result = FilteredAggregationFeatureGroup.match_feature_group_criteria(
            "value_int__sum_filtered_groupby", options, None
        )
        assert result is True

    def test_partition_by_must_be_list(self) -> None:
        """partition_by as a plain string (not a list) should fail validation."""
        options = Options(
            context={
                "partition_by": "region",
                "filter_column": "category",
                "filter_value": "X",
            }
        )
        result = FilteredAggregationFeatureGroup.match_feature_group_criteria(
            "value_int__sum_filtered_groupby", options, None
        )
        assert result is False

    def test_filter_column_required(self) -> None:
        """match_feature_group_criteria should fail without filter_column."""
        options = Options(context={"partition_by": ["region"], "filter_value": "X"})
        result = FilteredAggregationFeatureGroup.match_feature_group_criteria(
            "value_int__sum_filtered_groupby", options, None
        )
        assert result is False

    def test_filter_column_must_be_string(self) -> None:
        """filter_column must be a string, not an integer."""
        options = Options(
            context={
                "partition_by": ["region"],
                "filter_column": 42,
                "filter_value": "X",
            }
        )
        result = FilteredAggregationFeatureGroup.match_feature_group_criteria(
            "value_int__sum_filtered_groupby", options, None
        )
        assert result is False

    def test_filter_value_required(self) -> None:
        """match_feature_group_criteria should fail without filter_value."""
        options = Options(context={"partition_by": ["region"], "filter_column": "category"})
        result = FilteredAggregationFeatureGroup.match_feature_group_criteria(
            "value_int__sum_filtered_groupby", options, None
        )
        assert result is False

    def test_filter_value_accepts_string(self) -> None:
        """filter_value accepts string values."""
        result = FilteredAggregationFeatureGroup.match_feature_group_criteria(
            "value_int__sum_filtered_groupby", _VALID_OPTIONS, None
        )
        assert result is True

    def test_filter_value_accepts_integer(self) -> None:
        """filter_value accepts integer values."""
        options = Options(
            context={
                "partition_by": ["region"],
                "filter_column": "category",
                "filter_value": 42,
            }
        )
        result = FilteredAggregationFeatureGroup.match_feature_group_criteria(
            "value_int__sum_filtered_groupby", options, None
        )
        assert result is True

    def test_filter_value_accepts_float(self) -> None:
        """filter_value accepts float values."""
        options = Options(
            context={
                "partition_by": ["region"],
                "filter_column": "category",
                "filter_value": 3.14,
            }
        )
        result = FilteredAggregationFeatureGroup.match_feature_group_criteria(
            "value_int__sum_filtered_groupby", options, None
        )
        assert result is True

    def test_filter_value_accepts_boolean(self) -> None:
        """filter_value accepts boolean values."""
        options = Options(
            context={
                "partition_by": ["region"],
                "filter_column": "is_active",
                "filter_value": True,
            }
        )
        result = FilteredAggregationFeatureGroup.match_feature_group_criteria(
            "value_int__sum_filtered_groupby", options, None
        )
        assert result is True

    def test_filter_value_rejects_none(self) -> None:
        """filter_value must not be None."""
        options = Options(
            context={
                "partition_by": ["region"],
                "filter_column": "category",
                "filter_value": None,
            }
        )
        result = FilteredAggregationFeatureGroup.match_feature_group_criteria(
            "value_int__sum_filtered_groupby", options, None
        )
        assert result is False

    def test_filter_value_rejects_list(self) -> None:
        """filter_value must not be a list (single value only in v1)."""
        options = Options(
            context={
                "partition_by": ["region"],
                "filter_column": "category",
                "filter_value": ["X", "Y"],
            }
        )
        result = FilteredAggregationFeatureGroup.match_feature_group_criteria(
            "value_int__sum_filtered_groupby", options, None
        )
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
                "filter_column": "category",
                "filter_value": "X",
            }
        )
        result = FilteredAggregationFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is True

    def test_config_based_match_rejects_multiple_in_features(self) -> None:
        """Config-based feature with multiple in_features should not match (MAX_IN_FEATURES=1)."""
        from mloda.user import Feature as UserFeature

        options = Options(
            context={
                "aggregation_type": "sum",
                "in_features": frozenset({UserFeature("value_int"), UserFeature("value_float")}),
                "partition_by": ["region"],
                "filter_column": "category",
                "filter_value": "X",
            }
        )
        result = FilteredAggregationFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False

    def test_config_based_match_rejects_missing_partition_by(self) -> None:
        """Config-based feature without partition_by should not match."""
        options = Options(
            context={
                "aggregation_type": "sum",
                "in_features": "value_int",
                "filter_column": "category",
                "filter_value": "X",
            }
        )
        result = FilteredAggregationFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False

    def test_config_based_match_rejects_missing_filter_column(self) -> None:
        """Config-based feature without filter_column should not match."""
        options = Options(
            context={
                "aggregation_type": "sum",
                "in_features": "value_int",
                "partition_by": ["region"],
                "filter_value": "X",
            }
        )
        result = FilteredAggregationFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False

    def test_config_based_calculate_feature(self) -> None:
        """Config-based feature should compute correctly via calculate_feature."""
        import pyarrow as pa

        from mloda.core.abstract_plugins.components.feature_set import FeatureSet
        from mloda.community.feature_groups.data_operations.row_preserving.filtered_aggregation.pyarrow_filtered_aggregation import (
            PyArrowFilteredAggregation,
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
                    "filter_column": "category",
                    "filter_value": "X",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = PyArrowFilteredAggregation.calculate_feature(table, fs)
        assert isinstance(result, pa.Table)
        assert "my_sum_result" in result.column_names

        result_col = result.column("my_sum_result").to_pylist()
        expected = [10, 10, 10, 10, 60, 60, 60, 60, 15, 15, 15, -10]
        assert result_col == expected


class TestFilteredAggregationMatchValidation(MatchValidationTestBase):
    @classmethod
    def feature_group_class(cls) -> Any:
        return FilteredAggregationFeatureGroup

    @classmethod
    def valid_operations(cls) -> set[str]:
        return set(FilteredAggregationFeatureGroup.AGGREGATION_TYPES)

    @classmethod
    def config_key(cls) -> str:
        return "aggregation_type"

    @classmethod
    def build_feature_name(cls, operation: str) -> str:
        return f"value_int__{operation}_filtered_groupby"

    @classmethod
    def build_feature_name_no_source(cls) -> str:
        return "sum_filtered_groupby"

    @classmethod
    def additional_match_options(cls) -> dict[str, Any]:
        return {
            "in_features": "value_int",
            "partition_by": ["region"],
            "filter_column": "category",
            "filter_value": "X",
        }
