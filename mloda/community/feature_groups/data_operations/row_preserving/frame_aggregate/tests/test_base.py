"""Tests for FrameAggregateFeatureGroup base class."""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.match_validation import MatchValidationTestBase
from mloda.user import DataType, Feature

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.base import (
    FrameAggregateFeatureGroup,
    _AGGREGATION_TYPES,
)


class TestPatternParsing:
    """Tests for _parse_frame_feature pattern matching."""

    def test_rolling_pattern(self) -> None:
        result = FrameAggregateFeatureGroup._parse_frame_feature("sales__sum_rolling_3")
        assert result is not None
        assert result["source_col"] == "sales"
        assert result["agg_type"] == "sum"
        assert result["frame_type"] == "rolling"
        assert result["frame_size"] == 3
        assert result["frame_unit"] is None

    def test_time_window_pattern(self) -> None:
        result = FrameAggregateFeatureGroup._parse_frame_feature("price__avg_7_day_window")
        assert result is not None
        assert result["source_col"] == "price"
        assert result["agg_type"] == "avg"
        assert result["frame_type"] == "time"
        assert result["frame_size"] == 7
        assert result["frame_unit"] == "day"

    def test_cumulative_pattern(self) -> None:
        result = FrameAggregateFeatureGroup._parse_frame_feature("sales__cumsum")
        assert result is not None
        assert result["source_col"] == "sales"
        assert result["agg_type"] == "sum"
        assert result["frame_type"] == "cumulative"
        assert result["frame_size"] is None

    def test_expanding_pattern(self) -> None:
        result = FrameAggregateFeatureGroup._parse_frame_feature("score__expanding_avg")
        assert result is not None
        assert result["source_col"] == "score"
        assert result["agg_type"] == "avg"
        assert result["frame_type"] == "expanding"
        assert result["frame_size"] is None

    def test_no_match(self) -> None:
        result = FrameAggregateFeatureGroup._parse_frame_feature("plain_feature")
        assert result is None

    def test_rolling_large_window(self) -> None:
        result = FrameAggregateFeatureGroup._parse_frame_feature("value__max_rolling_100")
        assert result is not None
        assert result["frame_size"] == 100
        assert result["agg_type"] == "max"

    def test_time_window_hour(self) -> None:
        result = FrameAggregateFeatureGroup._parse_frame_feature("temp__min_24_hour_window")
        assert result is not None
        assert result["frame_unit"] == "hour"
        assert result["frame_size"] == 24

    def test_cummin(self) -> None:
        result = FrameAggregateFeatureGroup._parse_frame_feature("price__cummin")
        assert result is not None
        assert result["agg_type"] == "min"
        assert result["frame_type"] == "cumulative"

    def test_cummax(self) -> None:
        result = FrameAggregateFeatureGroup._parse_frame_feature("price__cummax")
        assert result is not None
        assert result["agg_type"] == "max"
        assert result["frame_type"] == "cumulative"

    def test_cumcount(self) -> None:
        result = FrameAggregateFeatureGroup._parse_frame_feature("price__cumcount")
        assert result is not None
        assert result["agg_type"] == "count"
        assert result["frame_type"] == "cumulative"


class TestParseFrameFeatureMemoization:
    """_parse_frame_feature must hand back fresh, independent dicts so callers cannot leak mutations.

    The lru_cache on ``_parse_frame_feature_cached`` shares one dict instance across calls; these
    guard that a mutation by one caller never reaches the next.
    """

    def test_mutating_result_does_not_leak(self) -> None:
        first = FrameAggregateFeatureGroup._parse_frame_feature("value__sum_rolling_3")
        assert first is not None
        first["agg_type"] = "mutated"
        second = FrameAggregateFeatureGroup._parse_frame_feature("value__sum_rolling_3")
        assert second is not None
        assert second["agg_type"] == "sum"

    def test_returns_distinct_objects(self) -> None:
        first = FrameAggregateFeatureGroup._parse_frame_feature("value__sum_rolling_3")
        second = FrameAggregateFeatureGroup._parse_frame_feature("value__sum_rolling_3")
        assert first is not second


class TestPatternMatching:
    """Tests for match_feature_group_criteria."""

    def _base_options(self) -> Options:
        return Options(context={"partition_by": ["region"], "order_by": "timestamp"})

    def test_matches_rolling_string(self) -> None:
        options = self._base_options()
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("sales__sum_rolling_3", options, None)
        assert result is True

    def test_matches_time_window_string(self) -> None:
        options = self._base_options()
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("sales__avg_7_day_window", options, None)
        assert result is True

    def test_matches_cumulative_string(self) -> None:
        options = self._base_options()
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("sales__cumsum", options, None)
        assert result is True

    def test_matches_expanding_string(self) -> None:
        options = self._base_options()
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("sales__expanding_avg", options, None)
        assert result is True

    def test_rejects_no_partition_by(self) -> None:
        options = Options(context={"order_by": "timestamp"})
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("sales__sum_rolling_3", options, None)
        assert result is False

    def test_rejects_no_order_by(self) -> None:
        options = Options(context={"partition_by": ["region"]})
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("sales__sum_rolling_3", options, None)
        assert result is False

    def test_rejects_invalid_agg_type(self) -> None:
        options = self._base_options()
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("sales__unknown_rolling_3", options, None)
        assert result is False

    def test_accepts_cumavg(self) -> None:
        """cumavg is a valid cumulative operation (cumulative and expanding are aliases)."""
        options = self._base_options()
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("sales__cumavg", options, None)
        assert result is True

    def test_rejects_invalid_time_unit(self) -> None:
        options = self._base_options()
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("sales__avg_7_banana_window", options, None)
        assert result is False

    def test_rejects_partition_by_as_string(self) -> None:
        options = Options(context={"partition_by": "region", "order_by": "timestamp"})
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("sales__sum_rolling_3", options, None)
        assert result is False

    def test_rejects_plain_feature_without_config(self) -> None:
        options = self._base_options()
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("plain_feature", options, None)
        assert result is False


class TestConfigBasedMatching:
    """Tests for configuration-based feature matching."""

    def test_config_rolling(self) -> None:
        options = Options(
            context={
                "aggregation_type": "sum",
                "frame_type": "rolling",
                "frame_size": 3,
                "in_features": "sales",
                "partition_by": ["region"],
                "order_by": "timestamp",
            }
        )
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is True

    def test_config_cumulative(self) -> None:
        options = Options(
            context={
                "aggregation_type": "sum",
                "frame_type": "cumulative",
                "in_features": "sales",
                "partition_by": ["region"],
                "order_by": "timestamp",
            }
        )
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is True

    def test_config_rejects_missing_agg_type(self) -> None:
        options = Options(
            context={
                "frame_type": "rolling",
                "partition_by": ["region"],
                "order_by": "timestamp",
            }
        )
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False

    def test_config_rejects_missing_frame_type(self) -> None:
        options = Options(
            context={
                "aggregation_type": "sum",
                "partition_by": ["region"],
                "order_by": "timestamp",
            }
        )
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False

    def test_config_rejects_invalid_frame_type(self) -> None:
        options = Options(
            context={
                "aggregation_type": "sum",
                "frame_type": "invalid",
                "partition_by": ["region"],
                "order_by": "timestamp",
            }
        )
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False

    def test_config_rejects_missing_in_features(self) -> None:
        # A config-based feature has no source column in its name, so in_features is
        # required; without it, extraction (get_in_features) would raise post-selection.
        options = Options(
            context={
                "aggregation_type": "count",
                "frame_type": "rolling",
                "frame_size": 3,
                "partition_by": ["region"],
                "order_by": "timestamp",
            }
        )
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("my_frame_agg", options, None)
        assert result is False

    def test_config_with_in_features_matches(self) -> None:
        # Regression guard: the same options plus in_features must match.
        options = Options(
            context={
                "aggregation_type": "count",
                "frame_type": "rolling",
                "frame_size": 3,
                "in_features": "sales",
                "partition_by": ["region"],
                "order_by": "timestamp",
            }
        )
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("my_frame_agg", options, None)
        assert result is True

    def test_name_based_matches_without_in_features(self) -> None:
        # The in_features requirement applies ONLY to the config path: a name-based
        # feature carries its source column in the name, so no in_features is needed.
        options = Options(context={"partition_by": ["region"], "order_by": "timestamp"})
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("sales__count_rolling_3", options, None)
        assert result is True


class TestConfigBasedFrameUnitValidation:
    """The config path must honour the declared frame_unit value space."""

    def _options(self, **overrides: Any) -> Options:
        context: dict[str, Any] = {
            "aggregation_type": "sum",
            "in_features": "sales",
            "partition_by": ["region"],
            "order_by": "timestamp",
        }
        context.update(overrides)
        return Options(context=context)

    def test_rejects_invalid_frame_unit(self) -> None:
        options = self._options(frame_type="time", frame_size=7, frame_unit="banana")
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False

    def test_rejects_missing_frame_unit(self) -> None:
        options = self._options(frame_type="time", frame_size=7)
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False

    def test_accepts_declared_frame_unit(self) -> None:
        options = self._options(frame_type="time", frame_size=7, frame_unit="day")
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is True


class TestConfigBasedFrameSizeValidation:
    """The config path must honour the declared frame_size value space.

    frame_size is a positive integer count; bools and numeric strings are not
    integers for this purpose and must be non-matches at discovery.
    """

    def _options(self, **overrides: Any) -> Options:
        context: dict[str, Any] = {
            "aggregation_type": "sum",
            "in_features": "sales",
            "partition_by": ["region"],
            "order_by": "timestamp",
        }
        context.update(overrides)
        return Options(context=context)

    @pytest.mark.parametrize("frame_size", [0, -1, True, "3"])
    def test_rejects_invalid_frame_size(self, frame_size: Any) -> None:
        options = self._options(frame_type="rolling", frame_size=frame_size)
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False, f"Config path should reject frame_size: {frame_size!r}"

    def test_accepts_positive_frame_size(self) -> None:
        options = self._options(frame_type="rolling", frame_size=3)
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is True

    def test_rejects_missing_frame_size_for_rolling(self) -> None:
        options = self._options(frame_type="rolling")
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False

    def test_rejects_missing_frame_size_for_time(self) -> None:
        options = self._options(frame_type="time", frame_unit="day")
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False

    @pytest.mark.parametrize("frame_type", ["cumulative", "expanding"])
    def test_accepts_missing_frame_size_for_unsized_frames(self, frame_type: str) -> None:
        options = self._options(frame_type=frame_type)
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is True, f"Config path should accept {frame_type} without frame_size"


class TestNameBasedZeroFrameSizeRejected:
    """A zero-sized frame is not a window, so the name path must reject it exactly as the config path does."""

    def _options(self) -> Options:
        return Options(context={"partition_by": ["region"], "order_by": "timestamp"})

    @pytest.mark.parametrize(
        "feature_name",
        [
            "sales__sum_rolling_0",
            "sales__avg_0_day_window",
            "sales__sum_rolling_00",
            "sales__avg_0_hour_window",
        ],
    )
    def test_rejects_zero_frame_size(self, feature_name: str) -> None:
        result = FrameAggregateFeatureGroup.match_feature_group_criteria(feature_name, self._options(), None)
        assert result is False, f"Name path should reject zero-sized frame: {feature_name}"

    @pytest.mark.parametrize("feature_name", ["sales__sum_rolling_0", "sales__avg_0_day_window"])
    def test_parse_returns_none_for_zero_frame_size(self, feature_name: str) -> None:
        assert FrameAggregateFeatureGroup._parse_frame_feature(feature_name) is None

    @pytest.mark.parametrize(
        "feature_name",
        ["sales__sum_rolling_3", "sales__avg_7_day_window", "sales__sum_rolling_10"],
    )
    def test_positive_frame_size_still_matches(self, feature_name: str) -> None:
        result = FrameAggregateFeatureGroup.match_feature_group_criteria(feature_name, self._options(), None)
        assert result is True, f"Name path should accept: {feature_name}"

    @pytest.mark.parametrize(
        "feature_name",
        ["sales__sum_rolling_3", "sales__avg_7_day_window", "sales__sum_rolling_10"],
    )
    def test_positive_frame_size_still_parses(self, feature_name: str) -> None:
        assert FrameAggregateFeatureGroup._parse_frame_feature(feature_name) is not None


class TestNameBasedUnaffectedByConfigValidation:
    """Name-based features carry their frame parameters in the name.

    Tightening the config path must not start demanding in_features or frame_size
    from a feature that encodes them in its name.
    """

    @pytest.mark.parametrize(
        "feature_name",
        [
            "sales__sum_rolling_3",
            "sales__avg_7_day_window",
            "sales__cumsum",
            "sales__expanding_avg",
        ],
    )
    def test_name_based_matches_with_partition_and_order_only(self, feature_name: str) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "timestamp"})
        result = FrameAggregateFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is True, f"Name path should accept: {feature_name}"


class TestNameBasedFrameTypeInOptions:
    """A frame name already encodes its size, so a frame_type carried in Options must not demand frame_size."""

    def test_rolling_name_with_frame_type_option(self) -> None:
        options = Options(context={"frame_type": "rolling", "partition_by": ["region"], "order_by": "timestamp"})
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("sales__sum_rolling_3", options, None)
        assert result is True

    def test_time_window_name_with_frame_type_option(self) -> None:
        options = Options(context={"frame_type": "time", "partition_by": ["region"], "order_by": "timestamp"})
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("sales__avg_7_day_window", options, None)
        assert result is True

    def test_rolling_name_with_propagated_frame_type(self) -> None:
        """A propagated frame_type must behave like a directly set one."""
        consumer = Options(context={"frame_type": "rolling"}, propagate_context_keys=frozenset({"frame_type"}))
        options = Options(context={"partition_by": ["region"], "order_by": "timestamp"})
        options.inherit_from(consumer)
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("sales__sum_rolling_3", options, None)
        assert result is True


class TestSingleTokenContainers:
    """A single-element container holds exactly one token, so it must reach dispatch unwrapped."""

    def _options(self, **overrides: Any) -> Options:
        context: dict[str, Any] = {
            "aggregation_type": "sum",
            "frame_type": "rolling",
            "frame_size": 3,
            "in_features": "sales",
            "partition_by": ["region"],
            "order_by": "timestamp",
        }
        context.update(overrides)
        return Options(context=context)

    @pytest.mark.parametrize("aggregation_type", [("sum",), ["sum"]])
    def test_single_element_aggregation_type(self, aggregation_type: Any) -> None:
        result = FrameAggregateFeatureGroup.match_feature_group_criteria(
            "my_result", self._options(aggregation_type=aggregation_type), None
        )
        assert result is True, f"Config path should accept: {aggregation_type!r}"
        feature = Feature("my_result", options=self._options(aggregation_type=aggregation_type))
        assert FrameAggregateFeatureGroup._extract_params(feature)["agg_type"] == "sum"

    @pytest.mark.parametrize("frame_type", [("rolling",), ["rolling"]])
    def test_single_element_frame_type(self, frame_type: Any) -> None:
        result = FrameAggregateFeatureGroup.match_feature_group_criteria(
            "my_result", self._options(frame_type=frame_type), None
        )
        assert result is True, f"Config path should accept: {frame_type!r}"
        feature = Feature("my_result", options=self._options(frame_type=frame_type))
        assert FrameAggregateFeatureGroup._extract_params(feature)["frame_type"] == "rolling"

    @pytest.mark.parametrize("frame_unit", [("day",), ["day"]])
    def test_single_element_frame_unit(self, frame_unit: Any) -> None:
        result = FrameAggregateFeatureGroup.match_feature_group_criteria(
            "my_result", self._options(frame_type="time", frame_size=7, frame_unit=frame_unit), None
        )
        assert result is True, f"Config path should accept: {frame_unit!r}"
        feature = Feature("my_result", options=self._options(frame_type="time", frame_size=7, frame_unit=frame_unit))
        assert FrameAggregateFeatureGroup._extract_params(feature)["frame_unit"] == "day"

    @pytest.mark.parametrize("aggregation_type", [["sum", "max"], ("sum", "max")])
    def test_multi_element_aggregation_type_rejected(self, aggregation_type: Any) -> None:
        result = FrameAggregateFeatureGroup.match_feature_group_criteria(
            "my_result", self._options(aggregation_type=aggregation_type), None
        )
        assert result is False, f"Config path should reject: {aggregation_type!r}"


class TestHostileInFeatures:
    """A hostile in_features value is a plain non-match; no exception may escape the matcher."""

    @pytest.mark.parametrize("in_features", ["", 0, 3.5, True, {"a": 1}, []])
    def test_rejects_hostile_in_features(self, in_features: Any) -> None:
        options = Options(
            context={
                "aggregation_type": "sum",
                "frame_type": "rolling",
                "frame_size": 3,
                "in_features": in_features,
                "partition_by": ["region"],
                "order_by": "timestamp",
            }
        )
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False, f"Config path should reject in_features: {in_features!r}"


class TestForwardedNameMismatch:
    """A forwarded aggregation_type that contradicts the name must raise, never be silently overridden."""

    def test_forwarded_aggregation_type_mismatch_raises(self) -> None:
        consumer = Options(group={"aggregation_type": "max"})
        options = Options(context={"partition_by": ["region"], "order_by": "timestamp"})
        options.inherit_from(consumer)
        with pytest.raises(ValueError):
            FrameAggregateFeatureGroup.match_feature_group_criteria("sales__sum_rolling_3", options, None)


#: Each frame name shape, its configuration equivalent, and the same shape with a malformed agg token.
_FRAME_SHAPE_PAIRS: list[tuple[str, dict[str, Any], str]] = [
    (
        "sales__sum_rolling_3",
        {"aggregation_type": "sum", "frame_type": "rolling", "frame_size": 3},
        "sales__banana_rolling_3",
    ),
    (
        "sales__avg_7_day_window",
        {"aggregation_type": "avg", "frame_type": "time", "frame_size": 7, "frame_unit": "day"},
        "sales__banana_7_day_window",
    ),
    (
        "sales__cumsum",
        {"aggregation_type": "sum", "frame_type": "cumulative"},
        "sales__cumbanana",
    ),
    (
        "sales__expanding_avg",
        {"aggregation_type": "avg", "frame_type": "expanding"},
        "sales__expanding_banana",
    ),
]


class TestFrameShapeParity:
    """All four name shapes, not only rolling, must agree with their config equivalent on both verdicts."""

    def _pattern_options(self) -> Options:
        return Options(context={"partition_by": ["region"], "order_by": "timestamp"})

    def _config_options(self, config: dict[str, Any]) -> Options:
        context: dict[str, Any] = {"in_features": "sales", "partition_by": ["region"], "order_by": "timestamp"}
        context.update(config)
        return Options(context=context)

    @pytest.mark.parametrize(("feature_name", "config", "malformed_name"), _FRAME_SHAPE_PAIRS)
    def test_shape_matches_on_both_paths(self, feature_name: str, config: dict[str, Any], malformed_name: str) -> None:
        by_name = FrameAggregateFeatureGroup.match_feature_group_criteria(feature_name, self._pattern_options(), None)
        assert by_name is True, f"Name path should accept: {feature_name}"
        by_config = FrameAggregateFeatureGroup.match_feature_group_criteria(
            "my_result", self._config_options(config), None
        )
        assert by_config is True, f"Config path should accept: {config}"

    @pytest.mark.parametrize(("feature_name", "config", "malformed_name"), _FRAME_SHAPE_PAIRS)
    def test_malformed_shape_rejected_on_both_paths(
        self, feature_name: str, config: dict[str, Any], malformed_name: str
    ) -> None:
        by_name = FrameAggregateFeatureGroup.match_feature_group_criteria(malformed_name, self._pattern_options(), None)
        assert by_name is False, f"Name path should reject: {malformed_name}"
        malformed_config = dict(config)
        malformed_config["aggregation_type"] = "banana"
        by_config = FrameAggregateFeatureGroup.match_feature_group_criteria(
            "my_result", self._config_options(malformed_config), None
        )
        assert by_config is False, f"Config path should reject: {malformed_config}"


class TestExtractParams:
    """Tests for _extract_params."""

    def test_extract_from_rolling_name(self) -> None:
        from mloda.user import Feature

        feature = Feature(
            "sales__sum_rolling_3",
            options=Options(context={"partition_by": ["region"], "order_by": "ts"}),
        )
        params = FrameAggregateFeatureGroup._extract_params(feature)
        assert params["source_col"] == "sales"
        assert params["agg_type"] == "sum"
        assert params["frame_type"] == "rolling"
        assert params["frame_size"] == 3
        assert params["partition_by"] == ["region"]
        assert params["order_by"] == "ts"

    def test_extract_from_config(self) -> None:
        from mloda.user import Feature

        feature = Feature(
            "my_result",
            options=Options(
                context={
                    "aggregation_type": "avg",
                    "frame_type": "expanding",
                    "in_features": "sales",
                    "partition_by": ["region"],
                    "order_by": "ts",
                }
            ),
        )
        params = FrameAggregateFeatureGroup._extract_params(feature)
        assert params["source_col"] == "sales"
        assert params["agg_type"] == "avg"
        assert params["frame_type"] == "expanding"
        assert params["partition_by"] == ["region"]


class TestReturnDataTypeRule:
    """return_data_type_rule should fix the output type only for deterministic ops.

    A rolling count always returns INT64. A rolling sum depends on the input
    column type, so the rule must return None for it.
    """

    def test_count_returns_int64(self) -> None:
        feature = Feature(
            "sales__count_rolling_3",
            options=Options(context={"partition_by": ["region"], "order_by": "timestamp"}),
        )
        assert FrameAggregateFeatureGroup.return_data_type_rule(feature) == DataType.INT64

    def test_sum_returns_none(self) -> None:
        feature = Feature(
            "sales__sum_rolling_3",
            options=Options(context={"partition_by": ["region"], "order_by": "timestamp"}),
        )
        assert FrameAggregateFeatureGroup.return_data_type_rule(feature) is None


class TestFrameAggregateMatchValidation(MatchValidationTestBase):
    @classmethod
    def feature_group_class(cls) -> Any:
        return FrameAggregateFeatureGroup

    @classmethod
    def valid_operations(cls) -> set[str]:
        return set(_AGGREGATION_TYPES)

    @classmethod
    def config_key(cls) -> str:
        return "aggregation_type"

    @classmethod
    def build_feature_name(cls, operation: str) -> str:
        return f"value_int__{operation}_rolling_3"

    @classmethod
    def build_feature_name_no_source(cls) -> str:
        return "sum_rolling_3"

    @classmethod
    def additional_match_options(cls) -> dict[str, Any]:
        return {
            "in_features": "value_int",
            "frame_type": "rolling",
            "frame_size": 3,
            "partition_by": ["region"],
            "order_by": "timestamp",
        }

    @classmethod
    def pattern_match_options(cls) -> Options:
        return Options(context={"partition_by": ["region"], "order_by": "timestamp"})

    @classmethod
    def parity_operations(cls) -> set[str]:
        # No parametric family here: the fixed aggregation types are the whole value space.
        return set(_AGGREGATION_TYPES)

    @classmethod
    def malformed_operations(cls) -> set[str]:
        return {"banana", "mode", "nunique", "first"}
