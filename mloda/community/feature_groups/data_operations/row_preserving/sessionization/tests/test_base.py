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

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.match_validation import ScalarArityTestBase, TokenCase
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


SESSIONIZE_FEATURE_NAME = "ts__sessionize_30_minute"


def _match_options() -> Options:
    """Options carrying the config the matcher requires."""
    return Options(context={"order_by": "ts", "partition_by": ["user"]})


class TestClassAttributes:
    def test_prefix_pattern_exists(self) -> None:
        assert hasattr(SessionizationFeatureGroup, "PREFIX_PATTERN")
        assert isinstance(SessionizationFeatureGroup.PREFIX_PATTERN, str)

    def test_prefix_pattern_value(self) -> None:
        assert SessionizationFeatureGroup.PREFIX_PATTERN == r".*__(sessionize_\d+_(?:minute|hour|day|week))$"

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

    @pytest.mark.parametrize(
        ("feature_name", "expected"),
        [
            pytest.param("x__sessionize_1_hour", True, id="n1_hour"),
            pytest.param("created_at__sessionize_15_minute", True, id="underscore_source"),
            pytest.param("ts__sessionize", False, id="no_unit"),
            pytest.param("ts__sessionize_30", False, id="missing_unit_token"),
            pytest.param("ts__sessionize_30_month", False, id="invalid_unit"),
            pytest.param("sessionize_30_minute", False, id="no_source_column"),
        ],
    )
    def test_match_by_name(self, feature_name: str, expected: bool) -> None:
        assert PandasSessionization.match_feature_group_criteria(feature_name, _match_options()) is expected

    def test_n_zero_matches_regex_but_rejected_at_parse(self) -> None:
        """``sessionize_0_minute`` matches the ``\\d+`` regex but n=0 is rejected at parse time.

        Mirrors ema's ``ema_0``: the pattern accepts the digit, and validation
        rejects n<=0 downstream (see TestThresholdParser).
        """
        # The regex itself accepts the digit 0.
        assert PandasSessionization.match_feature_group_criteria("ts__sessionize_0_minute", _match_options()) is True


class TestThresholdParser:
    @pytest.mark.parametrize(
        ("op_token", "expected"),
        [
            pytest.param("sessionize_30_minute", (30, "minute"), id="30_minute"),
            pytest.param("sessionize_1_hour", (1, "hour"), id="1_hour"),
            pytest.param("sessionize_2_day", (2, "day"), id="2_day"),
            pytest.param("sessionize_1_week", (1, "week"), id="1_week"),
        ],
    )
    def test_parse_op_components(self, op_token: str, expected: tuple[int, str]) -> None:
        assert _parse_sessionize_op(op_token) == expected

    @pytest.mark.parametrize(
        ("n", "unit", "expected"),
        [
            pytest.param(30, "minute", 30 * 60, id="minute"),
            pytest.param(1, "hour", 3600, id="hour"),
            pytest.param(2, "day", 2 * 86400, id="day"),
            pytest.param(1, "week", 604800, id="week"),
        ],
    )
    def test_threshold_seconds(self, n: int, unit: str, expected: int) -> None:
        assert _sessionize_threshold_seconds(n, unit) == expected

    @pytest.mark.parametrize(
        ("n", "unit", "match"),
        [
            pytest.param(1, "month", r"(?i)unit|month", id="bad_unit"),
            pytest.param(0, "minute", r"(?i)positive|> 0|n", id="n_zero"),
            pytest.param(-5, "minute", r"(?i)positive|> 0|n", id="negative_n"),
        ],
    )
    def test_threshold_seconds_rejects(self, n: int, unit: str, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            _sessionize_threshold_seconds(n, unit)

    def test_parse_rejects_bad_unit(self) -> None:
        with pytest.raises(ValueError, match=r"(?i)unit|month"):
            _parse_sessionize_op("sessionize_30_month")

    def test_parse_rejects_n_zero(self) -> None:
        with pytest.raises(ValueError, match=r"(?i)positive|> 0|n"):
            _parse_sessionize_op("sessionize_0_minute")


class TestOrderByArity(ScalarArityTestBase):
    """``order_by`` is a scalar key: one column, bare or in a single-element container.

    Sessionization declares no operation config key (the threshold is part of the feature
    name), so it declares the arity base directly instead of ``MatchValidationTestBase``.
    ``order_by`` uses a column DISTINCT from the source column, so an unwrap is
    distinguishable from the absent-value fallback to the source.
    """

    @classmethod
    def feature_group_class(cls) -> Any:
        return SessionizationFeatureGroup

    @classmethod
    def match_class(cls) -> Any:
        return PandasSessionization

    @classmethod
    def match_feature_name(cls) -> str:
        return SESSIONIZE_FEATURE_NAME

    @classmethod
    def base_context(cls) -> dict[str, Any]:
        return {"partition_by": ["user"]}

    @classmethod
    def token_cases(cls) -> list[TokenCase]:
        return [TokenCase("order_by", "event_ts", "user")]

    @classmethod
    def dispatch_values(cls, options: Options) -> list[Any]:
        feature = Feature(SESSIONIZE_FEATURE_NAME, options=options)
        return [SessionizationFeatureGroup._extract_order_by(feature, "ts")]

    def test_extract_order_by_defaults_to_source_column(self) -> None:
        """The one state the arity harness cannot express: absent is not wrapped, it is absent."""
        feature = Feature(SESSIONIZE_FEATURE_NAME, options=Options())
        assert SessionizationFeatureGroup._extract_order_by(feature, "ts") == "ts"


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
