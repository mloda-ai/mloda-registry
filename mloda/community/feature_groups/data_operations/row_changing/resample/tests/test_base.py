"""Tests for ResampleFeatureGroup base class."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.match_validation import MatchValidationTestBase, TokenCase
from mloda.user import DataType, Feature

from mloda.community.feature_groups.data_operations.row_changing.resample.base import (
    ResampleFeatureGroup,
)


class TestReturnDataTypeRule:
    """return_data_type_rule should fix the output type only for deterministic ops.

    A bucket count always returns INT64. mean / sum depend on the input column
    type, so the rule must return None for them.
    """

    def test_count_returns_int64(self) -> None:
        feature = Feature("value__resample_5_minute_count", options=Options())
        assert ResampleFeatureGroup.return_data_type_rule(feature) == DataType.INT64

    @pytest.mark.parametrize("agg", ["mean", "sum"])
    def test_input_dependent_ops_return_none(self, agg: str) -> None:
        feature = Feature(f"value__resample_5_minute_{agg}", options=Options())
        assert ResampleFeatureGroup.return_data_type_rule(feature) is None


class TestChainedNameParsing:
    """On a chained name, parsing must resolve the LAST ``resample`` marker, not the first."""

    CHAINED_SUM = "value__resample_1_hour_mean__resample_2_hour_sum"
    CHAINED_COUNT = "value__resample_1_hour_mean__resample_2_hour_count"

    def test_token_from_chained_name_uses_last_marker(self) -> None:
        feature = Feature(self.CHAINED_SUM, options=Options())
        assert ResampleFeatureGroup._extract_resample_op(feature) == "2_hour_sum"

    def test_source_from_chained_name_uses_last_marker(self) -> None:
        feature = Feature(self.CHAINED_SUM, options=Options())
        assert ResampleFeatureGroup._extract_source_features(feature) == ["value__resample_1_hour_mean"]

    def test_rule_reflects_last_token_count(self) -> None:
        # LAST token is a count -> deterministic INT64; must not raise.
        feature = Feature(self.CHAINED_COUNT, options=Options())
        assert ResampleFeatureGroup.return_data_type_rule(feature) == DataType.INT64

    def test_rule_reflects_last_token_sum(self) -> None:
        # LAST token is a sum -> input-dependent -> None; must not raise.
        feature = Feature(self.CHAINED_SUM, options=Options())
        assert ResampleFeatureGroup.return_data_type_rule(feature) is None


class TestPrefixPatternPerformance:
    """PREFIX_PATTERN must reject an adversarial name in bounded time.

    An earlier draft captured unit/agg with an ambiguous ``\\w+_\\w+`` (``_`` is
    itself a ``\\w`` character), which took ~22s of catastrophic backtracking on a
    ~6.4KB adversarial name. ``[a-z]+`` keeps unit/agg disjoint from the ``_``
    separator, so there is exactly one way to split the token and matching stays
    fast regardless of input length.
    """

    def test_adversarial_name_rejected_quickly(self) -> None:
        adversarial = "value" + ("__resample_1_a_b" * 400) + "!"
        options = Options(context={"time_column": "timestamp", "partition_by": ["region"]})

        start = time.monotonic()
        result = ResampleFeatureGroup.match_feature_group_criteria(adversarial, options, None)
        elapsed = time.monotonic() - start

        assert result is False
        assert elapsed < 2.0, f"match_feature_group_criteria took {elapsed:.3f}s on an adversarial name; expected < 2s"


class TestResampleOpConfig:
    """``resample_op`` participates in config-based selection and validation."""

    def test_resample_op_in_property_mapping(self) -> None:
        assert ResampleFeatureGroup.RESAMPLE_OP in ResampleFeatureGroup.PROPERTY_MAPPING


class TestOptionsPathInFeatureCount:
    """The options-based fallback in ``input_features`` must validate in_feature count like the name path."""

    def test_input_features_rejects_multiple_option_in_features(self) -> None:
        options = Options(
            context={
                "resample_op": "1_hour_mean",
                "time_column": "timestamp",
                "in_features": ["value_a", "value_b"],
            }
        )
        instance = ResampleFeatureGroup()
        with pytest.raises(ValueError, match="at most 1"):
            instance.input_features(options, FeatureName("my_result"))


class TestForwardedResampleOpMismatch:
    """A group-forwarded ``resample_op`` that contradicts the name-parsed op must be rejected, not silently ignored."""

    def test_mismatched_forwarded_resample_op_raises(self) -> None:
        consumer_options = Options(group={"resample_op": "2_hour_sum"})
        child_options = Options(context={"time_column": "timestamp"})
        child_options.inherit_from(consumer_options)

        with pytest.raises(ValueError, match="resample_op"):
            ResampleFeatureGroup.match_feature_group_criteria("value__resample_1_hour_mean", child_options, None)

    def test_matching_forwarded_resample_op_is_accepted(self) -> None:
        consumer_options = Options(group={"resample_op": "1_hour_mean"})
        child_options = Options(context={"time_column": "timestamp"})
        child_options.inherit_from(consumer_options)

        result = ResampleFeatureGroup.match_feature_group_criteria("value__resample_1_hour_mean", child_options, None)
        assert result is True


class TestResampleMatchValidation(MatchValidationTestBase):
    """Shared match-validation tests adapted for resample.

    The operation is the whole ``{n}_{unit}_{agg}`` token, on the name path as the
    suffix of ``__resample_`` and on the config path as ``resample_op``.
    """

    @classmethod
    def feature_group_class(cls) -> Any:
        return ResampleFeatureGroup

    @classmethod
    def valid_operations(cls) -> set[str]:
        return {"1_hour_mean", "15_minute_sum", "2_day_count", "1_hour_min", "3_hour_max"}

    @classmethod
    def config_key(cls) -> str:
        return "resample_op"

    @classmethod
    def build_feature_name(cls, operation: str) -> str:
        return f"value_int__resample_{operation}"

    @classmethod
    def build_feature_name_no_source(cls) -> str:
        return "resample_1_hour_mean"

    @classmethod
    def additional_match_options(cls) -> dict[str, Any]:
        return {"in_features": "value_int", "time_column": "timestamp", "partition_by": ["region"]}

    @classmethod
    def pattern_match_options(cls) -> Options:
        return Options(context={"time_column": "timestamp", "partition_by": ["region"]})

    @classmethod
    def malformed_operations(cls) -> set[str]:
        return {"0_hour_mean", "1_century_mean", "1_hour_median"}

    @classmethod
    def token_cases(cls) -> list[TokenCase]:
        # Unwrapping does not widen the value space: the token parser still owns resample_op.
        primary = replace(super().token_cases()[0], invalid=("not_a_token",))
        # time_column names one column, and resampling cannot pick a time axis without it.
        return [primary, TokenCase("time_column", "timestamp", "event_date", required=True)]

    @classmethod
    def dispatch_values(cls, options: Options) -> list[Any]:
        feature = Feature("my_result", options=options)
        return [
            ResampleFeatureGroup._extract_resample_op(feature),
            ResampleFeatureGroup._extract_time_column(feature),
        ]
