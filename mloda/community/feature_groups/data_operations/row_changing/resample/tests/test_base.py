"""Tests for ResampleFeatureGroup base class."""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.options import Options
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


class TestScalarKeyArity:
    """``time_column`` and ``resample_op`` are scalar keys: one value, bare or in a one-element container."""

    def _options(self, **overrides: Any) -> Options:
        context: dict[str, Any] = {
            "in_features": "value_int",
            "time_column": "timestamp",
            "resample_op": "1_hour_mean",
            "partition_by": ["region"],
        }
        context.update(overrides)
        return Options(context=context)

    @pytest.mark.parametrize("time_column", ["timestamp", ("timestamp",), ["timestamp"]])
    def test_singleton_time_column_matches(self, time_column: Any) -> None:
        result = ResampleFeatureGroup.match_feature_group_criteria(
            "my_result", self._options(time_column=time_column), None
        )
        assert result is True

    @pytest.mark.parametrize("time_column", [["timestamp", "region"], ("timestamp", "region"), 123])
    def test_invalid_time_column_rejected(self, time_column: Any) -> None:
        result = ResampleFeatureGroup.match_feature_group_criteria(
            "my_result", self._options(time_column=time_column), None
        )
        assert result is False

    @pytest.mark.parametrize("resample_op", ["1_hour_mean", ("1_hour_mean",), ["1_hour_mean"]])
    def test_singleton_resample_op_matches(self, resample_op: Any) -> None:
        result = ResampleFeatureGroup.match_feature_group_criteria(
            "my_result", self._options(resample_op=resample_op), None
        )
        assert result is True

    @pytest.mark.parametrize("resample_op", [["1_hour_mean", "1_day_sum"], ("not_a_token",)])
    def test_invalid_resample_op_rejected(self, resample_op: Any) -> None:
        """Unwrapping does not widen the value space: the token parser still owns it."""
        result = ResampleFeatureGroup.match_feature_group_criteria(
            "my_result", self._options(resample_op=resample_op), None
        )
        assert result is False

    @pytest.mark.parametrize("time_column", ["timestamp", ("timestamp",), ["timestamp"]])
    def test_extract_time_column_unwraps_to_bare_column(self, time_column: Any) -> None:
        feature = Feature("my_result", options=self._options(time_column=time_column))
        assert ResampleFeatureGroup._extract_time_column(feature) == "timestamp"

    def test_extract_time_column_raises_when_missing(self) -> None:
        feature = Feature("my_result", options=Options())
        with pytest.raises(ValueError, match="time_column"):
            ResampleFeatureGroup._extract_time_column(feature)

    @pytest.mark.parametrize("resample_op", ["1_hour_mean", ("1_hour_mean",), ["1_hour_mean"]])
    def test_extract_resample_op_unwraps_to_bare_token(self, resample_op: Any) -> None:
        feature = Feature("my_result", options=self._options(resample_op=resample_op))
        assert ResampleFeatureGroup._extract_resample_op(feature) == "1_hour_mean"
