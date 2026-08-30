"""Tests for AggregationFeatureGroup base class."""

from __future__ import annotations

from typing import Any

import pytest
from mloda.user import DataType, Feature, Options

from mloda.community.feature_groups.data_operations.aggregation.base import (
    AggregationFeatureGroup,
)
from mloda.testing.feature_groups.data_operations.match_validation import MatchValidationTestBase


class TestClassAttributes:
    """Tests for AggregationFeatureGroup class attributes."""

    def test_prefix_pattern_exists(self) -> None:
        """PREFIX_PATTERN regex attribute should be defined."""
        assert hasattr(AggregationFeatureGroup, "PREFIX_PATTERN")
        assert isinstance(AggregationFeatureGroup.PREFIX_PATTERN, str)

    def test_aggregation_types_exists(self) -> None:
        """AGGREGATION_TYPES dict should be defined with supported operations."""
        assert hasattr(AggregationFeatureGroup, "AGGREGATION_TYPES")
        assert isinstance(AggregationFeatureGroup.AGGREGATION_TYPES, dict)

    def test_aggregation_types_contains_standard_operations(self) -> None:
        """AGGREGATION_TYPES should contain standard aggregation operations."""
        expected_ops = {"sum", "avg", "count", "min", "max", "std", "var", "median"}
        for op in expected_ops:
            assert op in AggregationFeatureGroup.AGGREGATION_TYPES, f"Missing standard operation: {op}"

    def test_aggregation_types_contains_advanced_operations(self) -> None:
        """AGGREGATION_TYPES should contain advanced aggregation operations."""
        expected_ops = {"mode", "nunique", "first", "last"}
        for op in expected_ops:
            assert op in AggregationFeatureGroup.AGGREGATION_TYPES, f"Missing advanced operation: {op}"

    def test_min_in_features_is_one(self) -> None:
        """MIN_IN_FEATURES should be 1 (single source column)."""
        assert AggregationFeatureGroup.MIN_IN_FEATURES == 1

    def test_max_in_features_is_one(self) -> None:
        """MAX_IN_FEATURES should be 1 (single source column)."""
        assert AggregationFeatureGroup.MAX_IN_FEATURES == 1


class TestPatternMatching:
    """Tests for feature name pattern matching via match_feature_group_criteria."""

    @pytest.mark.parametrize(
        "feature_name",
        [
            "value_int__sum_agg",
            "value_int__avg_agg",
            "value_int__count_agg",
            "value_int__min_agg",
            "value_int__max_agg",
            "value_int__std_agg",
            "value_int__var_agg",
            "value_int__median_agg",
        ],
    )
    def test_matches_standard_operations(self, feature_name: str) -> None:
        """Standard aggregation operations with _agg suffix should match."""
        options = Options(context={"partition_by": ["region"]})
        result = AggregationFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is True, f"Should match: {feature_name}"

    @pytest.mark.parametrize(
        "feature_name",
        [
            "value_int__mode_agg",
            "value_int__nunique_agg",
            "value_int__first_agg",
            "value_int__last_agg",
        ],
    )
    def test_matches_advanced_operations(self, feature_name: str) -> None:
        """Advanced aggregation operations with _agg suffix should match."""
        options = Options(context={"partition_by": ["region"]})
        result = AggregationFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is True, f"Should match: {feature_name}"

    @pytest.mark.parametrize(
        "feature_name",
        [
            # percentile_75 parses as a dynamic type but has no implementation.
            pytest.param("value_int__percentile_75_agg", id="unimplemented_dynamic_type"),
            pytest.param("value_int__avg_window", id="wrong_suffix"),
            pytest.param("value_int__avg", id="no_suffix"),
            pytest.param("avg_agg", id="no_source_column"),
            pytest.param("value_int__unknown_agg", id="invalid_operation"),
        ],
    )
    def test_no_match(self, feature_name: str) -> None:
        options = Options(context={"partition_by": ["region"]})
        result = AggregationFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is False


class TestPatternParsing:
    """Tests for extracting operation and source column from feature names."""

    @pytest.mark.parametrize(
        ("feature_name", "expected"),
        [
            pytest.param("value_int__avg_agg", "avg", id="avg"),
            pytest.param("my_col__sum_agg", "sum", id="sum"),
            pytest.param("value_int__percentile_75_agg", "percentile_75", id="percentile"),
        ],
    )
    def test_parse_operation(self, feature_name: str, expected: str) -> None:
        operation = AggregationFeatureGroup.get_aggregation_type(feature_name)
        assert operation == expected

    def test_parse_source_feature_from_avg(self) -> None:
        """Source feature should be extracted correctly from value_int__avg_agg."""
        from mloda.user import Feature

        feature = Feature(
            "value_int__avg_agg",
            options=Options(context={"partition_by": ["region"]}),
        )
        source_features = AggregationFeatureGroup._extract_source_features(feature)
        assert source_features == ["value_int"]

    def test_parse_source_feature_from_sum(self) -> None:
        """Source feature should be extracted correctly from my_col__sum_agg."""
        from mloda.user import Feature

        feature = Feature(
            "my_col__sum_agg",
            options=Options(context={"partition_by": ["region"]}),
        )
        source_features = AggregationFeatureGroup._extract_source_features(feature)
        assert source_features == ["my_col"]


class TestPropertyMapping:
    """Tests for PROPERTY_MAPPING consistency."""

    def test_partition_by_in_property_mapping(self) -> None:
        """PARTITION_BY should be declared in PROPERTY_MAPPING for consistency with window aggregation."""
        assert AggregationFeatureGroup.PARTITION_BY in AggregationFeatureGroup.PROPERTY_MAPPING

    def test_partition_by_is_context_parameter(self) -> None:
        """PARTITION_BY should be declared as a context parameter."""
        mapping = AggregationFeatureGroup.PROPERTY_MAPPING[AggregationFeatureGroup.PARTITION_BY]
        assert mapping.context is True

    def test_property_mapping_has_aggregation_type(self) -> None:
        """AGGREGATION_TYPE should be in PROPERTY_MAPPING with strict validation."""
        mapping = AggregationFeatureGroup.PROPERTY_MAPPING[AggregationFeatureGroup.AGGREGATION_TYPE]
        assert mapping.strict_validation is True

    def test_property_mapping_has_in_features(self) -> None:
        """in_features should be in PROPERTY_MAPPING."""
        from mloda.provider import DefaultOptionKeys

        assert DefaultOptionKeys.in_features in AggregationFeatureGroup.PROPERTY_MAPPING


class TestConfigValidation:
    """Tests for partition_by configuration validation."""

    def test_partition_by_required(self) -> None:
        """match_feature_group_criteria should fail without partition_by in options."""
        options = Options(context={})
        result = AggregationFeatureGroup.match_feature_group_criteria("value_int__sum_agg", options, None)
        assert result is False

    def test_partition_by_accepts_list_of_strings(self) -> None:
        """partition_by should accept a list of strings."""
        options = Options(context={"partition_by": ["region", "country"]})
        result = AggregationFeatureGroup.match_feature_group_criteria("value_int__sum_agg", options, None)
        assert result is True

    def test_partition_by_accepts_tuple_of_strings(self) -> None:
        """partition_by should accept a tuple of strings (converted from list by mixin)."""
        options = Options(context={"partition_by": ("region", "country")})
        result = AggregationFeatureGroup.match_feature_group_criteria("value_int__sum_agg", options, None)
        assert result is True

    def test_partition_by_must_be_list_or_tuple(self) -> None:
        """partition_by as a plain string (not a list or tuple) should fail validation."""
        options = Options(context={"partition_by": "region"})
        result = AggregationFeatureGroup.match_feature_group_criteria("value_int__sum_agg", options, None)
        assert result is False

    def test_partition_by_rejects_non_string_items(self) -> None:
        """partition_by containing non-string items should fail validation."""
        options = Options(context={"partition_by": [123, "region"]})
        result = AggregationFeatureGroup.match_feature_group_criteria("value_int__sum_agg", options, None)
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
        result = AggregationFeatureGroup.match_feature_group_criteria("my_result", options, None)
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
        result = AggregationFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False

    def test_config_based_match_rejects_missing_partition_by(self) -> None:
        """Config-based feature without partition_by should not match."""
        options = Options(
            context={
                "aggregation_type": "sum",
                "in_features": "value_int",
            }
        )
        result = AggregationFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False

    def test_config_based_calculate_feature(self) -> None:
        """Config-based feature should compute correctly via calculate_feature."""
        import pyarrow as pa
        from mloda.provider import FeatureSet
        from mloda.user import Feature

        from mloda.community.feature_groups.data_operations.aggregation.pyarrow_aggregation import (
            PyArrowAggregation,
        )
        from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator

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

        result = PyArrowAggregation.calculate_feature(table, fs)
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


class TestReturnDataTypeRule:
    """return_data_type_rule should fix the output type only for deterministic ops.

    count/nunique always return INT64 regardless of input. sum/avg depend on
    the input column type, so the rule must return None for them.
    """

    @pytest.mark.parametrize("operation", ["count", "nunique"])
    def test_deterministic_ops_return_int64(self, operation: str) -> None:
        feature = Feature(
            f"value_int__{operation}_agg",
            options=Options(context={"partition_by": ["region"]}),
        )
        assert AggregationFeatureGroup.return_data_type_rule(feature) == DataType.INT64

    @pytest.mark.parametrize("operation", ["sum", "avg"])
    def test_input_dependent_ops_return_none(self, operation: str) -> None:
        feature = Feature(
            f"value_int__{operation}_agg",
            options=Options(context={"partition_by": ["region"]}),
        )
        assert AggregationFeatureGroup.return_data_type_rule(feature) is None


class TestAggregationMatchValidation(MatchValidationTestBase):
    @classmethod
    def feature_group_class(cls) -> Any:
        return AggregationFeatureGroup

    @classmethod
    def valid_operations(cls) -> set[str]:
        return set(AggregationFeatureGroup.AGGREGATION_TYPES)

    @classmethod
    def config_key(cls) -> str:
        return "aggregation_type"

    @classmethod
    def build_feature_name(cls, operation: str) -> str:
        return f"value_int__{operation}_agg"

    @classmethod
    def build_feature_name_no_source(cls) -> str:
        return "sum_agg"

    @classmethod
    def additional_match_options(cls) -> dict[str, Any]:
        return {"in_features": "value_int", "partition_by": ["region"]}

    @classmethod
    def pattern_match_options(cls) -> Options:
        return Options(context={"partition_by": ["region"]})

    @classmethod
    def dispatch_values(cls, options: Options) -> list[Any]:
        feature = Feature("my_result", options=options)
        return [
            AggregationFeatureGroup._extract_aggregation_type(feature),
            AggregationFeatureGroup._resolve_agg_type("my_result", options),
        ]
