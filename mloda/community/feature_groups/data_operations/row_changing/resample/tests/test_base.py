"""Tests for ResampleFeatureGroup base class."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

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
    """Token / source parsing must agree with the anchored (end-matched) pattern.

    The PREFIX_PATTERN is anchored at the end, so on a chained name the operative
    token is the LAST ``resample`` segment and the source is everything before it.
    ``_token_from_name`` / ``_source_from_name`` must split on the LAST marker.
    """

    CHAINED_SUM = "value__resample_1_hour_mean__resample_2_hour_sum"
    CHAINED_COUNT = "value__resample_1_hour_mean__resample_2_hour_count"

    def test_token_from_chained_name_uses_last_marker(self) -> None:
        assert ResampleFeatureGroup._token_from_name(self.CHAINED_SUM) == "2_hour_sum"

    def test_source_from_chained_name_uses_last_marker(self) -> None:
        assert ResampleFeatureGroup._source_from_name(self.CHAINED_SUM) == "value__resample_1_hour_mean"

    def test_rule_reflects_last_token_count(self) -> None:
        # LAST token is a count -> deterministic INT64; must not raise.
        feature = Feature(self.CHAINED_COUNT, options=Options())
        assert ResampleFeatureGroup.return_data_type_rule(feature) == DataType.INT64

    def test_rule_reflects_last_token_sum(self) -> None:
        # LAST token is a sum -> input-dependent -> None; must not raise.
        feature = Feature(self.CHAINED_SUM, options=Options())
        assert ResampleFeatureGroup.return_data_type_rule(feature) is None


class TestResampleOpConfig:
    """``resample_op`` participates in config-based selection and validation."""

    def test_resample_op_in_property_mapping(self) -> None:
        assert ResampleFeatureGroup.RESAMPLE_OP in ResampleFeatureGroup.PROPERTY_MAPPING


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
