"""Tests for BinningFeatureGroup base class."""

from __future__ import annotations

from typing import Any

import pytest
from mloda.user import DataType, Feature, Options

from mloda.community.feature_groups.data_operations.row_preserving.binning.base import (
    BINNING_OPS,
    BinningFeatureGroup,
)
from mloda.testing.feature_groups.data_operations.match_validation import MatchValidationTestBase, TokenCase


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

    @pytest.mark.parametrize(
        "feature_name",
        [
            pytest.param("value_int__bucket_3", id="wrong_suffix"),
            pytest.param("value_int__bin", id="no_number"),
            pytest.param("bin_3", id="no_source_column"),
        ],
    )
    def test_no_match(self, feature_name: str) -> None:
        options = Options()
        result = BinningFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is False

    @pytest.mark.parametrize(
        "feature_name",
        [
            "value_int__bin_0",
            "value_int__qbin_0",
        ],
    )
    def test_no_match_zero_bins(self, feature_name: str) -> None:
        # n_bins must be >= 1, so the prefix pattern rejects a trailing 0.
        options = Options()
        result = BinningFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is False, f"Should not match (n_bins must be >= 1): {feature_name}"


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
        # bin_0 no longer matches the prefix pattern (n_bins must be >= 1), so the
        # parameters cannot be extracted at all.
        with pytest.raises(ValueError, match="Could not extract binning parameters"):
            BinningFeatureGroup.get_binning_params("value_int__bin_0")

    def test_n_bins_zero_qbin_raises(self) -> None:
        with pytest.raises(ValueError, match="Could not extract binning parameters"):
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

    @pytest.mark.parametrize("invalid_n_bins", [0, -1, "abc"])
    def test_config_based_match_rejects_invalid_n_bins(self, invalid_n_bins: Any) -> None:
        # n_bins must be a positive integer; 0, negatives and non-ints must not match.
        options = Options(
            context={
                "binning_op": "bin",
                "n_bins": invalid_n_bins,
                "in_features": "value_int",
            }
        )
        result = BinningFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False


class TestNBinsCoercion:
    """The ``n_bins`` check the arity harness cannot express.

    Every arity verdict for ``n_bins`` (bare, wrapped, multi-element, wrong type and the
    values outside its value space) is declared on ``TestBinningMatchValidation``; what
    stays here is the return type a direct extractor call owes its caller.
    """

    @pytest.mark.parametrize("n_bins", [5, (5,), 5.9, (5.9,)])
    def test_extract_binning_params_returns_an_int(self, n_bins: Any) -> None:
        """The signature says ``tuple[str, int]``: a direct call must coerce, not just unwrap.

        ``is_positive_int`` keeps a float out at match time, so only a direct call can
        reach here with one, and it must still hand the backend an int bin count.
        """
        options = Options(context={"binning_op": "bin", "n_bins": n_bins, "in_features": "value_int"})
        _, extracted = BinningFeatureGroup._extract_binning_params(Feature("my_result", options=options))
        assert extracted == 5
        assert isinstance(extracted, int)


class TestReturnDataTypeRule:
    """return_data_type_rule should fix the output type for deterministic ops.

    Both bin and qbin emit integer bin indices, so the rule returns INT64.
    """

    def test_bin_returns_int64(self) -> None:
        feature = Feature("value_int__bin_5", options=Options())
        assert BinningFeatureGroup.return_data_type_rule(feature) == DataType.INT64

    def test_qbin_returns_int64(self) -> None:
        feature = Feature("value_int__qbin_4", options=Options())
        assert BinningFeatureGroup.return_data_type_rule(feature) == DataType.INT64


class TestForwardedNBinsMismatch:
    """A group-forwarded ``n_bins`` that contradicts the name-parsed value is rejected, not silently ignored."""

    def test_mismatched_forwarded_n_bins_raises(self) -> None:
        consumer_options = Options(group={"n_bins": 10})
        child_options = Options()
        child_options.inherit_from(consumer_options)

        with pytest.raises(ValueError, match="n_bins"):
            BinningFeatureGroup.match_feature_group_criteria("value_int__bin_5", child_options, None)

    def test_matching_forwarded_n_bins_is_accepted(self) -> None:
        consumer_options = Options(group={"n_bins": 5})
        child_options = Options()
        child_options.inherit_from(consumer_options)

        result = BinningFeatureGroup.match_feature_group_criteria("value_int__bin_5", child_options, None)
        assert result is True

    def test_locally_set_contradictory_n_bins_is_not_flagged(self) -> None:
        options = Options(context={"n_bins": 10})

        result = BinningFeatureGroup.match_feature_group_criteria("value_int__bin_5", options, None)
        assert result is True

    def test_mismatched_n_bins_env_downgrade_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MLODA_ALLOW_FORWARDED_NAME_MISMATCH", "1")
        consumer_options = Options(group={"n_bins": 10})
        child_options = Options()
        child_options.inherit_from(consumer_options)

        result = BinningFeatureGroup.match_feature_group_criteria("value_int__bin_5", child_options, None)
        assert result is True


class TestBinningMatchValidation(MatchValidationTestBase):
    @classmethod
    def feature_group_class(cls) -> Any:
        return BinningFeatureGroup

    @classmethod
    def valid_operations(cls) -> set[str]:
        return set(BINNING_OPS)

    @classmethod
    def config_key(cls) -> str:
        return "binning_op"

    @classmethod
    def build_feature_name(cls, operation: str) -> str:
        return f"value_int__{operation}_5"

    @classmethod
    def build_feature_name_no_source(cls) -> str:
        return "equal_width_5"

    @classmethod
    def additional_match_options(cls) -> dict[str, Any]:
        return {"in_features": "value_int", "n_bins": 5}

    @classmethod
    def token_cases(cls) -> list[TokenCase]:
        # n_bins is scalar too: one positive int, so zero, a bool and a digit string stay out.
        return [*super().token_cases(), TokenCase("n_bins", 5, 10, invalid=(0, True, "5"))]

    @classmethod
    def dispatch_values(cls, options: Options) -> list[Any]:
        return list(BinningFeatureGroup._extract_binning_params(Feature("my_result", options=options)))
