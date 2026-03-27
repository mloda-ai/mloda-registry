"""Tests for BinningFeatureGroup base class."""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.components.options import Options

from mloda.community.feature_groups.data_operations.row_preserving.binning.base import (
    BINNING_OPS,
    BinningFeatureGroup,
)


class TestClassAttributes:
    def test_prefix_pattern_exists(self) -> None:
        assert hasattr(BinningFeatureGroup, "PREFIX_PATTERN")
        assert isinstance(BinningFeatureGroup.PREFIX_PATTERN, str)

    def test_binning_ops_contains_all_operations(self) -> None:
        expected_ops = {"bin", "qbin"}
        for op in expected_ops:
            assert op in BINNING_OPS, f"Missing operation: {op}"

    def test_min_in_features_is_one(self) -> None:
        assert BinningFeatureGroup.MIN_IN_FEATURES == 1

    def test_max_in_features_is_one(self) -> None:
        assert BinningFeatureGroup.MAX_IN_FEATURES == 1


class TestPatternMatching:
    @pytest.mark.parametrize(
        "feature_name",
        [
            "value_int__bin_3",
            "value_int__bin_5",
            "value_int__bin_10",
            "value_float__bin_4",
            "value_int__qbin_3",
            "value_int__qbin_5",
            "value_float__qbin_4",
        ],
    )
    def test_matches_valid_binning_features(self, feature_name: str) -> None:
        options = Options()
        result = BinningFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is True, f"Should match: {feature_name}"

    def test_no_match_wrong_suffix(self) -> None:
        options = Options()
        result = BinningFeatureGroup.match_feature_group_criteria("value_int__bucket_3", options, None)
        assert result is False

    def test_no_match_no_number(self) -> None:
        options = Options()
        result = BinningFeatureGroup.match_feature_group_criteria("value_int__bin", options, None)
        assert result is False

    def test_no_match_no_source_column(self) -> None:
        options = Options()
        result = BinningFeatureGroup.match_feature_group_criteria("bin_3", options, None)
        assert result is False


class TestPatternParsing:
    def test_parse_bin_operation(self) -> None:
        op, n_bins = BinningFeatureGroup.get_binning_params("value_int__bin_5")
        assert op == "bin"
        assert n_bins == 5

    def test_parse_qbin_operation(self) -> None:
        op, n_bins = BinningFeatureGroup.get_binning_params("value_int__qbin_10")
        assert op == "qbin"
        assert n_bins == 10

    def test_n_bins_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            BinningFeatureGroup.get_binning_params("value_int__bin_0")

    def test_n_bins_zero_qbin_raises(self) -> None:
        with pytest.raises(ValueError, match="n_bins must be >= 1"):
            BinningFeatureGroup.get_binning_params("value_int__qbin_0")

    def test_parse_source_feature(self) -> None:
        from mloda.user import Feature

        feature = Feature("value_int__bin_3", options=Options())
        source_features = BinningFeatureGroup._extract_source_features(feature)
        assert source_features == ["value_int"]

    def test_parse_source_feature_with_underscores(self) -> None:
        from mloda.user import Feature

        feature = Feature("my_value_int__bin_5", options=Options())
        source_features = BinningFeatureGroup._extract_source_features(feature)
        assert source_features == ["my_value_int"]


class TestConfigBasedFeatures:
    def test_config_based_match(self) -> None:
        options = Options(
            context={
                "binning_op": "bin",
                "n_bins": 5,
                "in_features": "value_int",
            }
        )
        result = BinningFeatureGroup.match_feature_group_criteria("my_binned_result", options, None)
        assert result is True

    def test_config_based_match_rejects_invalid_op(self) -> None:
        options = Options(
            context={
                "binning_op": "invalid_op",
                "n_bins": 5,
                "in_features": "value_int",
            }
        )
        result = BinningFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False
