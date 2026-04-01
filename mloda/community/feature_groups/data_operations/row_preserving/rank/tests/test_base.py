"""Tests for RankFeatureGroup base class."""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import DefaultOptionKeys
from mloda.testing.feature_groups.data_operations.match_validation import MatchValidationTestBase

from mloda.community.feature_groups.data_operations.row_preserving.rank.base import (
    RankFeatureGroup,
)


class TestClassAttributes:
    """Tests for RankFeatureGroup class attributes."""

    def test_prefix_pattern_exists(self) -> None:
        assert hasattr(RankFeatureGroup, "PREFIX_PATTERN")
        assert isinstance(RankFeatureGroup.PREFIX_PATTERN, str)

    def test_rank_types_exists(self) -> None:
        assert hasattr(RankFeatureGroup, "RANK_TYPES")
        assert isinstance(RankFeatureGroup.RANK_TYPES, dict)

    def test_rank_types_contains_standard_operations(self) -> None:
        expected_ops = {"row_number", "rank", "dense_rank", "percent_rank"}
        for op in expected_ops:
            assert op in RankFeatureGroup.RANK_TYPES, f"Missing rank type: {op}"

    def test_supports_ntile(self) -> None:
        assert RankFeatureGroup._supports_rank_type("ntile_4")
        assert RankFeatureGroup._supports_rank_type("ntile_10")

    def test_rejects_invalid_ntile(self) -> None:
        assert not RankFeatureGroup._supports_rank_type("ntile_0")
        assert not RankFeatureGroup._supports_rank_type("ntile_abc")

    def test_supports_ntile_1(self) -> None:
        assert RankFeatureGroup._supports_rank_type("ntile_1")

    def test_rejects_ntile_negative(self) -> None:
        assert not RankFeatureGroup._supports_rank_type("ntile_-1")

    def test_supports_top_n(self) -> None:
        assert RankFeatureGroup._supports_rank_type("top_5")
        assert RankFeatureGroup._supports_rank_type("top_1")

    def test_supports_bottom_n(self) -> None:
        assert RankFeatureGroup._supports_rank_type("bottom_5")
        assert RankFeatureGroup._supports_rank_type("bottom_1")

    def test_rejects_invalid_top_n(self) -> None:
        assert not RankFeatureGroup._supports_rank_type("top_0")
        assert not RankFeatureGroup._supports_rank_type("top_abc")

    def test_rejects_invalid_bottom_n(self) -> None:
        assert not RankFeatureGroup._supports_rank_type("bottom_0")
        assert not RankFeatureGroup._supports_rank_type("bottom_abc")

    def test_min_in_features_is_one(self) -> None:
        assert RankFeatureGroup.MIN_IN_FEATURES == 1

    def test_max_in_features_is_one(self) -> None:
        assert RankFeatureGroup.MAX_IN_FEATURES == 1


class TestPropertyMapping:
    """Tests for PROPERTY_MAPPING consistency with context parameter discovery."""

    def test_property_mapping_contains_partition_by(self) -> None:
        mapping = RankFeatureGroup.PROPERTY_MAPPING
        assert RankFeatureGroup.PARTITION_BY in mapping
        entry = mapping[RankFeatureGroup.PARTITION_BY]
        assert entry[DefaultOptionKeys.context] is True

    def test_property_mapping_contains_order_by(self) -> None:
        mapping = RankFeatureGroup.PROPERTY_MAPPING
        assert RankFeatureGroup.ORDER_BY in mapping
        entry = mapping[RankFeatureGroup.ORDER_BY]
        assert entry[DefaultOptionKeys.context] is True

    def test_property_mapping_contains_rank_type(self) -> None:
        mapping = RankFeatureGroup.PROPERTY_MAPPING
        assert RankFeatureGroup.RANK_TYPE in mapping
        entry = mapping[RankFeatureGroup.RANK_TYPE]
        assert entry[DefaultOptionKeys.context] is True

    def test_property_mapping_contains_in_features(self) -> None:
        mapping = RankFeatureGroup.PROPERTY_MAPPING
        assert DefaultOptionKeys.in_features in mapping
        entry = mapping[DefaultOptionKeys.in_features]
        assert entry[DefaultOptionKeys.context] is True


class TestPatternMatching:
    """Tests for feature name pattern matching."""

    @pytest.mark.parametrize(
        "feature_name",
        [
            "value_int__row_number_ranked",
            "value_int__rank_ranked",
            "value_int__dense_rank_ranked",
            "value_int__percent_rank_ranked",
        ],
    )
    def test_matches_standard_rank_types(self, feature_name: str) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "value_int"})
        result = RankFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is True, f"Should match: {feature_name}"

    def test_matches_ntile(self) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "value_int"})
        result = RankFeatureGroup.match_feature_group_criteria("value_int__ntile_4_ranked", options, None)
        assert result is True

    def test_no_match_wrong_suffix(self) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "value_int"})
        result = RankFeatureGroup.match_feature_group_criteria("value_int__rank_window", options, None)
        assert result is False

    def test_no_match_no_source_column(self) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "value_int"})
        result = RankFeatureGroup.match_feature_group_criteria("rank_ranked", options, None)
        assert result is False

    def test_no_match_invalid_rank_type(self) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "value_int"})
        result = RankFeatureGroup.match_feature_group_criteria("value_int__unknown_ranked", options, None)
        assert result is False

    def test_matches_top_n(self) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "value_int"})
        result = RankFeatureGroup.match_feature_group_criteria("value_int__top_5_ranked", options, None)
        assert result is True

    def test_matches_bottom_n(self) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "value_int"})
        result = RankFeatureGroup.match_feature_group_criteria("value_int__bottom_3_ranked", options, None)
        assert result is True


class TestPatternParsing:
    """Tests for extracting rank type and source column."""

    def test_parse_row_number(self) -> None:
        rank_type = RankFeatureGroup.get_rank_type("value_int__row_number_ranked")
        assert rank_type == "row_number"

    def test_parse_dense_rank(self) -> None:
        rank_type = RankFeatureGroup.get_rank_type("my_col__dense_rank_ranked")
        assert rank_type == "dense_rank"

    def test_parse_ntile(self) -> None:
        rank_type = RankFeatureGroup.get_rank_type("value_int__ntile_4_ranked")
        assert rank_type == "ntile_4"

    def test_parse_top_n(self) -> None:
        rank_type = RankFeatureGroup.get_rank_type("value_int__top_5_ranked")
        assert rank_type == "top_5"

    def test_parse_bottom_n(self) -> None:
        rank_type = RankFeatureGroup.get_rank_type("value_int__bottom_3_ranked")
        assert rank_type == "bottom_3"

    def test_parse_source_feature(self) -> None:
        from mloda.user import Feature

        feature = Feature(
            "value_int__rank_ranked",
            options=Options(context={"partition_by": ["region"], "order_by": "value_int"}),
        )
        source_features = RankFeatureGroup._extract_source_features(feature)
        assert source_features == ["value_int"]


class TestConfigValidation:
    """Tests for partition_by and order_by validation."""

    def test_partition_by_required(self) -> None:
        options = Options(context={"order_by": "value_int"})
        result = RankFeatureGroup.match_feature_group_criteria("value_int__rank_ranked", options, None)
        assert result is False

    def test_order_by_required(self) -> None:
        options = Options(context={"partition_by": ["region"]})
        result = RankFeatureGroup.match_feature_group_criteria("value_int__rank_ranked", options, None)
        assert result is False

    def test_partition_by_must_be_list_or_tuple(self) -> None:
        options = Options(context={"partition_by": "region", "order_by": "value_int"})
        result = RankFeatureGroup.match_feature_group_criteria("value_int__rank_ranked", options, None)
        assert result is False

    def test_partition_by_accepts_tuple(self) -> None:
        options = Options(context={"partition_by": ("region",), "order_by": "value_int"})
        result = RankFeatureGroup.match_feature_group_criteria("value_int__rank_ranked", options, None)
        assert result is True

    def test_partition_by_rejects_empty_list(self) -> None:
        options = Options(context={"partition_by": [], "order_by": "value_int"})
        result = RankFeatureGroup.match_feature_group_criteria("value_int__rank_ranked", options, None)
        assert result is False

    def test_partition_by_rejects_empty_tuple(self) -> None:
        options = Options(context={"partition_by": (), "order_by": "value_int"})
        result = RankFeatureGroup.match_feature_group_criteria("value_int__rank_ranked", options, None)
        assert result is False

    def test_order_by_must_be_string(self) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": ["value_int"]})
        result = RankFeatureGroup.match_feature_group_criteria("value_int__rank_ranked", options, None)
        assert result is False

    def test_valid_config(self) -> None:
        options = Options(context={"partition_by": ["region", "category"], "order_by": "value_int"})
        result = RankFeatureGroup.match_feature_group_criteria("value_int__rank_ranked", options, None)
        assert result is True


class TestConfigBasedFeatures:
    """Tests for configuration-based feature matching."""

    def test_config_based_match(self) -> None:
        options = Options(
            context={
                "rank_type": "row_number",
                "in_features": "value_int",
                "partition_by": ["region"],
                "order_by": "value_int",
            }
        )
        result = RankFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is True

    def test_config_based_match_rejects_missing_order_by(self) -> None:
        options = Options(
            context={
                "rank_type": "row_number",
                "in_features": "value_int",
                "partition_by": ["region"],
            }
        )
        result = RankFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False

    def test_config_based_calculate_feature(self) -> None:
        import pyarrow as pa

        from mloda.core.abstract_plugins.components.feature_set import FeatureSet
        from mloda.community.feature_groups.data_operations.row_preserving.rank.pyarrow_rank import PyArrowRank
        from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
        from mloda.user import Feature

        table = PyArrowDataOpsTestDataCreator.create()

        feature = Feature(
            "my_rank_result",
            options=Options(
                context={
                    "rank_type": "row_number",
                    "in_features": "value_int",
                    "partition_by": ["region"],
                    "order_by": "value_int",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = PyArrowRank.calculate_feature(table, fs)
        assert isinstance(result, pa.Table)
        assert "my_rank_result" in result.column_names
        assert result.num_rows == 12


class TestRankMatchValidation(MatchValidationTestBase):
    @classmethod
    def feature_group_class(cls) -> Any:
        return RankFeatureGroup

    @classmethod
    def valid_operations(cls) -> set[str]:
        return set(RankFeatureGroup.RANK_TYPES)

    @classmethod
    def config_key(cls) -> str:
        return "rank_type"

    @classmethod
    def build_feature_name(cls, operation: str) -> str:
        return f"value_int__{operation}_ranked"

    @classmethod
    def build_feature_name_no_source(cls) -> str:
        return "row_number_ranked"

    @classmethod
    def additional_match_options(cls) -> dict[str, Any]:
        return {"in_features": "value_int", "partition_by": ["region"], "order_by": "value_int"}
