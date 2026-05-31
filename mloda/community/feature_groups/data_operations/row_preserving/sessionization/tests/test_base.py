"""Tests for SessionizationFeatureGroup base class.

Covers the ``{ts}__sessionize_{n}_{unit}`` grammar: pattern matching (valid
names accepted, invalid names rejected), the threshold parser
(``n``/``unit`` -> seconds), and single-source-column enforcement.

The matcher reads ``partition_by`` / ``order_by`` from the Options context, so
match tests pass an Options carrying that config (mirroring ema's
``TestEmaMatchFeatureGroupCriteria``). ``ts__sessionize_0_minute`` MATCHES the
regex (``\\d+`` accepts ``0``) but is rejected at parse/compute time (mirroring
ema's ``ema_0`` handling).
"""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.row_preserving.sessionization.base import (
    SESSIONIZATION_UNITS,
    SessionizationFeatureGroup,
    _parse_sessionize_op,
    _sessionize_threshold_seconds,
)
from mloda.community.feature_groups.data_operations.row_preserving.sessionization.pandas_sessionization import (
    PandasSessionization,
)


def _match_options() -> Options:
    """Options carrying the config the matcher requires."""
    return Options(context={"order_by": "ts", "partition_by": ["user"]})


class TestClassAttributes:
    def test_prefix_pattern_exists(self) -> None:
        assert hasattr(SessionizationFeatureGroup, "PREFIX_PATTERN")
        assert isinstance(SessionizationFeatureGroup.PREFIX_PATTERN, str)

    def test_prefix_pattern_value(self) -> None:
        assert SessionizationFeatureGroup.PREFIX_PATTERN == r".*__sessionize_\d+_(?:minute|hour|day|week)$"

    def test_sessionization_units_contains_four_units(self) -> None:
        assert set(SESSIONIZATION_UNITS.keys()) == {"minute", "hour", "day", "week"}

    def test_min_in_features_is_one(self) -> None:
        assert SessionizationFeatureGroup.MIN_IN_FEATURES == 1

    def test_max_in_features_is_one(self) -> None:
        assert SessionizationFeatureGroup.MAX_IN_FEATURES == 1


class TestPatternMatching:
    @pytest.mark.parametrize("unit", ["minute", "hour", "day", "week"])
    def test_matches_each_unit(self, unit: str) -> None:
        name = f"ts__sessionize_30_{unit}"
        assert PandasSessionization.match_feature_group_criteria(name, _match_options()) is True

    def test_matches_n1_hour(self) -> None:
        assert PandasSessionization.match_feature_group_criteria("x__sessionize_1_hour", _match_options()) is True

    def test_matches_underscore_source(self) -> None:
        assert (
            PandasSessionization.match_feature_group_criteria("created_at__sessionize_15_minute", _match_options())
            is True
        )

    def test_no_match_no_unit(self) -> None:
        assert PandasSessionization.match_feature_group_criteria("ts__sessionize", _match_options()) is False

    def test_no_match_missing_unit_token(self) -> None:
        assert PandasSessionization.match_feature_group_criteria("ts__sessionize_30", _match_options()) is False

    def test_no_match_invalid_unit(self) -> None:
        assert PandasSessionization.match_feature_group_criteria("ts__sessionize_30_month", _match_options()) is False

    def test_no_match_no_source_column(self) -> None:
        assert PandasSessionization.match_feature_group_criteria("sessionize_30_minute", _match_options()) is False

    def test_n_zero_matches_regex_but_rejected_at_parse(self) -> None:
        """``sessionize_0_minute`` matches the ``\\d+`` regex but n=0 is rejected at parse time.

        Mirrors ema's ``ema_0``: the pattern accepts the digit, and validation
        rejects n<=0 downstream (see TestThresholdParser).
        """
        # The regex itself accepts the digit 0.
        assert PandasSessionization.match_feature_group_criteria("ts__sessionize_0_minute", _match_options()) is True


class TestThresholdParser:
    def test_parse_30_minute(self) -> None:
        assert _parse_sessionize_op("sessionize_30_minute") == (30, "minute")

    def test_parse_1_hour(self) -> None:
        assert _parse_sessionize_op("sessionize_1_hour") == (1, "hour")

    def test_parse_2_day(self) -> None:
        assert _parse_sessionize_op("sessionize_2_day") == (2, "day")

    def test_parse_1_week(self) -> None:
        assert _parse_sessionize_op("sessionize_1_week") == (1, "week")

    def test_threshold_seconds_minute(self) -> None:
        assert _sessionize_threshold_seconds(30, "minute") == 30 * 60

    def test_threshold_seconds_hour(self) -> None:
        assert _sessionize_threshold_seconds(1, "hour") == 3600

    def test_threshold_seconds_day(self) -> None:
        assert _sessionize_threshold_seconds(2, "day") == 2 * 86400

    def test_threshold_seconds_week(self) -> None:
        assert _sessionize_threshold_seconds(1, "week") == 604800

    def test_threshold_seconds_rejects_bad_unit(self) -> None:
        with pytest.raises(ValueError, match=r"(?i)unit|month"):
            _sessionize_threshold_seconds(1, "month")

    def test_threshold_seconds_rejects_n_zero(self) -> None:
        with pytest.raises(ValueError, match=r"(?i)positive|> 0|n"):
            _sessionize_threshold_seconds(0, "minute")

    def test_threshold_seconds_rejects_negative_n(self) -> None:
        with pytest.raises(ValueError, match=r"(?i)positive|> 0|n"):
            _sessionize_threshold_seconds(-5, "minute")

    def test_parse_rejects_bad_unit(self) -> None:
        with pytest.raises(ValueError, match=r"(?i)unit|month"):
            _parse_sessionize_op("sessionize_30_month")

    def test_parse_rejects_n_zero(self) -> None:
        with pytest.raises(ValueError, match=r"(?i)positive|> 0|n"):
            _parse_sessionize_op("sessionize_0_minute")


class TestSingleColumnEnforcement:
    def test_input_features_rejects_multiple_option_in_features(self) -> None:
        options = Options(
            context={
                "in_features": ["ts_a", "ts_b"],
                "partition_by": ["user"],
                "order_by": "ts_a",
            }
        )
        instance = SessionizationFeatureGroup()
        with pytest.raises(ValueError, match="at most 1"):
            instance.input_features(options, FeatureName("my_result"))

    def test_extract_source_features_rejects_multiple_in_features(self) -> None:
        options = Options(
            context={
                "in_features": ["ts_a", "ts_b"],
                "partition_by": ["user"],
                "order_by": "ts_a",
            }
        )
        feature = Feature("my_result", options=options)
        with pytest.raises(ValueError, match="at most 1"):
            SessionizationFeatureGroup._extract_source_features(feature)

    def test_extract_source_features_returns_single_item_for_string_pattern(self) -> None:
        feature = Feature("ts__sessionize_30_minute", options=Options())
        source_features = SessionizationFeatureGroup._extract_source_features(feature)
        assert source_features == ["ts"]

    def test_extract_source_features_with_underscores(self) -> None:
        feature = Feature("created_at__sessionize_30_minute", options=Options())
        source_features = SessionizationFeatureGroup._extract_source_features(feature)
        assert source_features == ["created_at"]
