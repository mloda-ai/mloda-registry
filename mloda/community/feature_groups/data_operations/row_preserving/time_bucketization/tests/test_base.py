"""Tests for TimeBucketizationFeatureGroup base class."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.match_validation import MatchValidationTestBase
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.row_preserving.time_bucketization.base import (
    TIME_BUCKETIZATION_OPS,
    TIME_BUCKETIZATION_UNITS,
    TimeBucketizationFeatureGroup,
    _parse_bucket_op,
)


# Helper builders for parametrized tests.
def _valid_n1_tokens() -> list[str]:
    """The 18 universally-valid (op, n=1, unit) tokens across all 6 units."""
    return [f"{op}_1_{unit}" for op in ("floor", "ceil", "round") for unit in TIME_BUCKETIZATION_UNITS]


def _valid_multi_n_tokens() -> list[str]:
    """A small selection of n>1 tokens that are valid (only fixed-freq units)."""
    return [
        "floor_5_minute",
        "ceil_15_minute",
        "round_30_minute",
        "floor_2_hour",
        "ceil_6_hour",
        "round_2_day",
    ]


class TestClassAttributes:
    def test_prefix_pattern_exists(self) -> None:
        assert hasattr(TimeBucketizationFeatureGroup, "PREFIX_PATTERN")
        assert isinstance(TimeBucketizationFeatureGroup.PREFIX_PATTERN, str)

    def test_time_bucketization_ops_contains_three_ops(self) -> None:
        assert set(TIME_BUCKETIZATION_OPS.keys()) == {"floor", "ceil", "round"}

    def test_time_bucketization_units_contains_six_units(self) -> None:
        assert set(TIME_BUCKETIZATION_UNITS.keys()) == {"minute", "hour", "day", "week", "month", "year"}

    def test_min_in_features_is_one(self) -> None:
        assert TimeBucketizationFeatureGroup.MIN_IN_FEATURES == 1

    def test_max_in_features_is_one(self) -> None:
        assert TimeBucketizationFeatureGroup.MAX_IN_FEATURES == 1

    def test_bucket_op_constant(self) -> None:
        assert TimeBucketizationFeatureGroup.BUCKET_OP == "bucket_op"


class TestPatternMatching:
    @pytest.mark.parametrize("op_token", _valid_n1_tokens())
    def test_matches_all_n1_tokens(self, op_token: str) -> None:
        feature_name = f"timestamp__{op_token}"
        options = Options()
        result = TimeBucketizationFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is True, f"Should match: {feature_name}"

    @pytest.mark.parametrize("op_token", _valid_multi_n_tokens())
    def test_matches_n_gt_1_fixed_freq_tokens(self, op_token: str) -> None:
        feature_name = f"timestamp__{op_token}"
        options = Options()
        result = TimeBucketizationFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is True, f"Should match: {feature_name}"

    def test_no_match_wrong_suffix(self) -> None:
        options = Options()
        result = TimeBucketizationFeatureGroup.match_feature_group_criteria("timestamp__truncate_1_day", options, None)
        assert result is False

    def test_no_match_no_suffix(self) -> None:
        options = Options()
        result = TimeBucketizationFeatureGroup.match_feature_group_criteria("timestamp", options, None)
        assert result is False

    def test_no_match_no_source_column(self) -> None:
        options = Options()
        # ``floor_1_day`` alone has no source-column prefix.
        result = TimeBucketizationFeatureGroup.match_feature_group_criteria("floor_1_day", options, None)
        assert result is False

    def test_no_match_invalid_op(self) -> None:
        options = Options()
        result = TimeBucketizationFeatureGroup.match_feature_group_criteria("timestamp__truncate_1_day", options, None)
        assert result is False

    def test_no_match_invalid_unit(self) -> None:
        options = Options()
        result = TimeBucketizationFeatureGroup.match_feature_group_criteria("timestamp__floor_1_century", options, None)
        assert result is False

    def test_no_match_n_zero(self) -> None:
        """``floor_0_day`` is grammatically valid for the regex (``\\d+``) but ``n=0``
        is rejected by validation. The pattern accepts; the FG validation rejects."""
        options = Options()
        result = TimeBucketizationFeatureGroup.match_feature_group_criteria("timestamp__floor_0_day", options, None)
        # Pattern itself allows ``\d+`` to include 0; the FG's _validate_string_match
        # must reject n=0 so this match returns False.
        assert result is False


class TestPatternParsing:
    def test_parse_floor_1_day(self) -> None:
        op = TimeBucketizationFeatureGroup.get_bucket_op("timestamp__floor_1_day")
        assert op == "floor_1_day"

    def test_parse_ceil_5_minute(self) -> None:
        op = TimeBucketizationFeatureGroup.get_bucket_op("timestamp__ceil_5_minute")
        assert op == "ceil_5_minute"

    def test_parse_round_1_week(self) -> None:
        op = TimeBucketizationFeatureGroup.get_bucket_op("timestamp__round_1_week")
        assert op == "round_1_week"

    def test_parse_op_components_floor_1_day(self) -> None:
        """Module-level helper splits the op token into (op, n, unit)."""
        components = _parse_bucket_op("floor_1_day")
        assert components == ("floor", 1, "day")

    def test_parse_op_components_ceil_15_minute(self) -> None:
        components = _parse_bucket_op("ceil_15_minute")
        assert components == ("ceil", 15, "minute")

    def test_parse_op_components_round_1_year(self) -> None:
        components = _parse_bucket_op("round_1_year")
        assert components == ("round", 1, "year")

    def test_parse_source_feature(self) -> None:
        feature = Feature("timestamp__floor_1_day", options=Options())
        source_features = TimeBucketizationFeatureGroup._extract_source_features(feature)
        assert source_features == ["timestamp"]

    def test_parse_source_feature_with_underscores(self) -> None:
        feature = Feature("created_at__floor_1_day", options=Options())
        source_features = TimeBucketizationFeatureGroup._extract_source_features(feature)
        assert source_features == ["created_at"]

    def test_greedy_regex_for_chained_op_tokens(self) -> None:
        """Pin the greedy-parse contract shared with sibling families.

        ``rsplit("__", 1)`` plus a greedy ``.*__(...)$`` pattern means for a
        chained name like ``my_event__hour__floor_1_day`` the source is
        ``my_event__hour`` and the captured op token is ``floor_1_day``.

        This mirrors scalar_arithmetic's pinned contract; a future regex
        tightening must be a deliberate decision and this test surfaces any
        silent change.
        """
        feature = Feature("my_event__hour__floor_1_day", options=Options())
        assert TimeBucketizationFeatureGroup._extract_source_features(feature) == ["my_event__hour"]
        assert TimeBucketizationFeatureGroup._extract_bucket_op(feature) == "floor_1_day"


class TestConfigBasedFeatures:
    def test_config_based_match(self) -> None:
        options = Options(
            context={
                "bucket_op": "floor_1_day",
                "in_features": "timestamp",
            }
        )
        result = TimeBucketizationFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is True

    def test_config_based_match_rejects_invalid_op(self) -> None:
        options = Options(
            context={
                "bucket_op": "bogus_1_day",
                "in_features": "timestamp",
            }
        )
        result = TimeBucketizationFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False

    def test_config_based_match_rejects_invalid_unit(self) -> None:
        options = Options(
            context={
                "bucket_op": "floor_1_century",
                "in_features": "timestamp",
            }
        )
        result = TimeBucketizationFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False

    def test_config_based_match_rejects_n_zero(self) -> None:
        options = Options(
            context={
                "bucket_op": "floor_0_day",
                "in_features": "timestamp",
            }
        )
        result = TimeBucketizationFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False


class TestSingleColumnEnforcement:
    """Verify that MAX_IN_FEATURES=1 enforces single-column behavior."""

    def test_max_in_features_is_one(self) -> None:
        assert TimeBucketizationFeatureGroup.MAX_IN_FEATURES == 1

    def test_input_features_rejects_multiple_option_in_features(self) -> None:
        options = Options(
            context={
                "bucket_op": "floor_1_day",
                "in_features": ["timestamp_a", "timestamp_b"],
            }
        )
        instance = TimeBucketizationFeatureGroup()
        with pytest.raises(ValueError, match="at most 1"):
            instance.input_features(options, FeatureName("my_result"))

    def test_extract_source_features_rejects_multiple_in_features(self) -> None:
        options = Options(
            context={
                "bucket_op": "floor_1_day",
                "in_features": ["timestamp_a", "timestamp_b"],
            }
        )
        feature = Feature("my_result", options=options)
        with pytest.raises(ValueError, match="at most 1"):
            TimeBucketizationFeatureGroup._extract_source_features(feature)

    def test_extract_source_features_rejects_empty_in_features(self) -> None:
        """Empty in_features must raise ValueError at the FG-level guard.

        ``Options.get_in_features()`` itself rejects empty values today, so
        end-to-end the user sees a ValueError. But the FG-level guard in
        ``_extract_source_features`` does not enforce ``MIN_IN_FEATURES``:
        if a caller hands back an empty frozenset (e.g. a future Options
        relaxation, a custom Options subclass, or a non-Options path), the
        function silently returns ``[]`` and ``calculate_feature`` then does
        ``source_features[0]`` and raises a bare ``IndexError``.

        We bypass the Options-layer guard with a mock to exercise the FG
        contract directly: empty in_features must raise ValueError with the
        same "at most"-style message as the multi-column case.
        """
        feature = Feature(
            "my_result",
            options=Options(
                context={
                    "bucket_op": "floor_1_day",
                    "in_features": "placeholder",
                }
            ),
        )
        with patch.object(feature.options, "get_in_features", return_value=frozenset()):
            with pytest.raises(ValueError, match=r"(?i)at least|source|in_features"):
                TimeBucketizationFeatureGroup._extract_source_features(feature)

    def test_extract_source_features_returns_single_item_for_string_pattern(self) -> None:
        feature = Feature("timestamp__floor_1_day", options=Options())
        source_features = TimeBucketizationFeatureGroup._extract_source_features(feature)
        assert len(source_features) == 1
        assert source_features == ["timestamp"]

    def test_extract_source_features_returns_single_item_for_option_config(self) -> None:
        options = Options(
            context={
                "bucket_op": "ceil_1_day",
                "in_features": "event_time",
            }
        )
        feature = Feature("my_result", options=options)
        source_features = TimeBucketizationFeatureGroup._extract_source_features(feature)
        assert len(source_features) == 1
        assert source_features == ["event_time"]

    def test_input_features_returns_single_feature_for_string_pattern(self) -> None:
        instance = TimeBucketizationFeatureGroup()
        result = instance.input_features(Options(), FeatureName("timestamp__floor_1_day"))
        assert result is not None
        assert len(result) == 1
        assert {f.name for f in result} == {"timestamp"}

    def test_input_features_returns_single_feature_for_option_config(self) -> None:
        options = Options(
            context={
                "bucket_op": "floor_1_day",
                "in_features": "event_time",
            }
        )
        instance = TimeBucketizationFeatureGroup()
        result = instance.input_features(options, FeatureName("my_result"))
        assert result is not None
        assert len(result) == 1
        assert {f.name for f in result} == {"event_time"}


class TestBucketOpExtraction:
    """Verify bucket-op extraction from both string and option sources."""

    def test_get_bucket_op_raises_for_non_pattern_name(self) -> None:
        with pytest.raises(ValueError, match="Could not extract"):
            TimeBucketizationFeatureGroup.get_bucket_op("plain_name")

    def test_extract_bucket_op_from_options(self) -> None:
        options = Options(
            context={
                "bucket_op": "ceil_5_minute",
                "in_features": "timestamp",
            }
        )
        feature = Feature("my_result", options=options)
        op = TimeBucketizationFeatureGroup._extract_bucket_op(feature)
        assert op == "ceil_5_minute"

    def test_extract_bucket_op_raises_without_option(self) -> None:
        feature = Feature("plain_name", options=Options())
        with pytest.raises(ValueError, match="Could not extract"):
            TimeBucketizationFeatureGroup._extract_bucket_op(feature)

    @pytest.mark.parametrize("op_token", _valid_n1_tokens())
    def test_get_bucket_op_for_all_n1_tokens(self, op_token: str) -> None:
        feature_name = f"col__{op_token}"
        result = TimeBucketizationFeatureGroup.get_bucket_op(feature_name)
        assert result == op_token


class TestTimeBucketizationMatchValidation(MatchValidationTestBase):
    """Shared match-validation tests adapted for time bucketization.

    ``valid_operations()`` returns the 18 universally-valid n=1 tokens (3 ops
    x 6 units). All are valid on every backend; n>1 carve-outs for
    week/month/year do not need to appear here because the harness uses
    representative samples (``next(iter(...))``) plus iteration for case and
    special-char tests.
    """

    @classmethod
    def feature_group_class(cls) -> Any:
        return TimeBucketizationFeatureGroup

    @classmethod
    def valid_operations(cls) -> set[str]:
        return set(_valid_n1_tokens())

    @classmethod
    def config_key(cls) -> str:
        return "bucket_op"

    @classmethod
    def build_feature_name(cls, operation: str) -> str:
        return f"timestamp__{operation}"

    @classmethod
    def build_feature_name_no_source(cls) -> str:
        return "floor_1_day"

    @classmethod
    def additional_match_options(cls) -> dict[str, Any]:
        return {"in_features": "timestamp"}
