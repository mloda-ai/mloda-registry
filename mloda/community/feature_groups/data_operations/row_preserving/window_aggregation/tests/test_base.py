"""Tests for WindowAggregationFeatureGroup base class."""

from __future__ import annotations

from typing import Any

import pytest
from mloda.user import DataType, Feature, Options

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
    WindowAggregationFeatureGroup,
)
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.helpers import extract_column, feature_set_for
from mloda.testing.feature_groups.data_operations.match_validation import MatchValidationTestBase, TokenCase


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
            "value_int__sum_window",
            "value_int__avg_window",
            "value_int__count_window",
            "value_int__min_window",
            "value_int__max_window",
            "value_int__std_window",
            "value_int__var_window",
            "value_int__median_window",
        ],
    )
    def test_matches_standard_operations(self, feature_name: str) -> None:
        """Standard aggregation operations with _window suffix should match."""
        options = Options(context={"partition_by": ["region"]})
        result = WindowAggregationFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is True, f"Should match: {feature_name}"

    @pytest.mark.parametrize(
        "feature_name",
        [
            "value_int__mode_window",
            "value_int__nunique_window",
        ],
    )
    def test_matches_advanced_operations(self, feature_name: str) -> None:
        """Advanced aggregation operations with _window suffix should match."""
        options = Options(context={"partition_by": ["region"]})
        result = WindowAggregationFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is True, f"Should match: {feature_name}"

    @pytest.mark.parametrize(
        "feature_name",
        [
            pytest.param("value_int__avg_grouped", id="wrong_suffix"),
            pytest.param("value_int__avg", id="no_suffix"),
            pytest.param("avg_window", id="no_source_column"),
            pytest.param("value_int__unknown_window", id="invalid_operation"),
        ],
    )
    def test_no_match(self, feature_name: str) -> None:
        options = Options(context={"partition_by": ["region"]})
        result = WindowAggregationFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is False


class TestPatternParsing:
    """Tests for extracting operation and source column from feature names."""

    def test_parse_avg_operation(self) -> None:
        """Parsing value_int__avg_window should yield operation=avg, source=value_int."""
        operation = WindowAggregationFeatureGroup.get_aggregation_type("value_int__avg_window")
        assert operation == "avg"

    def test_parse_sum_operation(self) -> None:
        """Parsing my_col__sum_window should yield operation=sum, source=my_col."""
        operation = WindowAggregationFeatureGroup.get_aggregation_type("my_col__sum_window")
        assert operation == "sum"

    def test_parse_source_feature_from_avg(self) -> None:
        """Source feature should be extracted correctly from value_int__avg_window."""
        from mloda.user import Feature

        feature = Feature(
            "value_int__avg_window",
            options=Options(context={"partition_by": ["region"]}),
        )
        source_features = WindowAggregationFeatureGroup._extract_source_features(feature)
        assert source_features == ["value_int"]

    def test_parse_source_feature_from_sum(self) -> None:
        """Source feature should be extracted correctly from my_col__sum_window."""
        from mloda.user import Feature

        feature = Feature(
            "my_col__sum_window",
            options=Options(context={"partition_by": ["region"]}),
        )
        source_features = WindowAggregationFeatureGroup._extract_source_features(feature)
        assert source_features == ["my_col"]


class TestConfigValidation:
    """Tests for partition_by configuration validation."""

    @pytest.mark.parametrize(
        ("feature_name", "context", "expected"),
        [
            pytest.param("value_int__sum_window", {}, False, id="partition_by_required"),
            pytest.param(
                "value_int__sum_window",
                {"partition_by": ["region", "country"]},
                True,
                id="partition_by_accepts_list_of_strings",
            ),
            pytest.param("value_int__sum_window", {"partition_by": "region"}, False, id="partition_by_must_be_list"),
            pytest.param("value_int__first_window", {"partition_by": ["region"]}, False, id="first_requires_order_by"),
            pytest.param("value_int__last_window", {"partition_by": ["region"]}, False, id="last_requires_order_by"),
            pytest.param(
                "value_int__first_window",
                {"partition_by": ["region"], "order_by": "value_int"},
                True,
                id="first_matches_with_order_by",
            ),
            # sum is order-independent
            pytest.param(
                "value_int__sum_window", {"partition_by": ["region"]}, True, id="sum_does_not_require_order_by"
            ),
        ],
    )
    def test_match_feature_group_criteria(self, feature_name: str, context: dict[str, Any], expected: bool) -> None:
        # Options keeps the dict it is handed and mutates it on other paths, so the shared
        # argvalue must not be passed in directly.
        options = Options(context=dict(context))
        result = WindowAggregationFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is expected

    def test_partition_by_rejects_empty_list(self) -> None:
        options = Options(context={"partition_by": []})
        assert not WindowAggregationFeatureGroup.match_feature_group_criteria("value_int__sum_window", options, None)

    def test_partition_by_rejects_empty_tuple(self) -> None:
        options = Options(context={"partition_by": ()})
        assert not WindowAggregationFeatureGroup.match_feature_group_criteria("value_int__sum_window", options, None)


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
        from mloda.provider import FeatureSet
        from mloda.user import Feature

        from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.pyarrow_window_aggregation import (
            PyArrowWindowAggregation,
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

        result = PyArrowWindowAggregation.calculate_feature(table, fs)
        assert isinstance(result, pa.Table)
        assert "my_sum_result" in result.column_names

        result_col = result.column("my_sum_result").to_pylist()
        expected = [25, 25, 25, 25, 140, 140, 140, 140, 70, 70, 70, -10]
        assert result_col == expected


class TestReturnDataTypeRule:
    """return_data_type_rule should fix the output type only for deterministic ops.

    count/nunique always return INT64. avg depends on the input column type, so
    the rule must return None for it.
    """

    @pytest.mark.parametrize("operation", ["count", "nunique"])
    def test_deterministic_ops_return_int64(self, operation: str) -> None:
        feature = Feature(
            f"value_int__{operation}_window",
            options=Options(context={"partition_by": ["region"]}),
        )
        assert WindowAggregationFeatureGroup.return_data_type_rule(feature) == DataType.INT64

    def test_avg_returns_none(self) -> None:
        feature = Feature(
            "value_int__avg_window",
            options=Options(context={"partition_by": ["region"]}),
        )
        assert WindowAggregationFeatureGroup.return_data_type_rule(feature) is None


class TestWindowAggregationMatchValidation(MatchValidationTestBase):
    @classmethod
    def feature_group_class(cls) -> Any:
        return WindowAggregationFeatureGroup

    @classmethod
    def valid_operations(cls) -> set[str]:
        return set(WindowAggregationFeatureGroup.AGGREGATION_TYPES)

    @classmethod
    def config_key(cls) -> str:
        return "aggregation_type"

    @classmethod
    def build_feature_name(cls, operation: str) -> str:
        return f"value_int__{operation}_window"

    @classmethod
    def build_feature_name_no_source(cls) -> str:
        return "sum_window"

    @classmethod
    def additional_match_options(cls) -> dict[str, Any]:
        # order_by is required for first/last and harmless for the other types.
        return {"in_features": "value_int", "partition_by": ["region"], "order_by": "timestamp"}

    @classmethod
    def pattern_match_options(cls) -> Options:
        return Options(context={"partition_by": ["region"], "order_by": "timestamp"})

    @classmethod
    def token_cases(cls) -> list[TokenCase]:
        # first/last are the aggregation types that require order_by.
        return [
            *super().token_cases(),
            TokenCase(cls.config_key(), "first", without=("order_by",), matches=False),
            TokenCase(cls.config_key(), "first"),
            # order_by names one column; declare it under the agg type that requires it.
            TokenCase("order_by", "timestamp", "region", context={cls.config_key(): "first"}),
        ]

    @classmethod
    def dispatch_values(cls, options: Options) -> list[Any]:
        feature = Feature("my_result", options=options)
        return [
            WindowAggregationFeatureGroup._extract_aggregation_type(feature),
            WindowAggregationFeatureGroup._resolve_agg_type("my_result", options),
        ]

    @classmethod
    def compute_values(cls, options: Options) -> list[Any] | None:
        # Window aggregation reads order_by inline in calculate_feature rather than through an
        # extractor, so only a run through the backend shows a container reaching it unwrapped.
        from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.pyarrow_window_aggregation import (
            PyArrowWindowAggregation,
        )

        result = PyArrowWindowAggregation.calculate_feature(
            PyArrowDataOpsTestDataCreator.create(), feature_set_for("my_window_result", options)
        )
        return extract_column(result, "my_window_result")
