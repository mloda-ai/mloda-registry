"""Tests for FrameAggregateFeatureGroup base class."""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import Any

import pytest
from mloda.provider import (
    FeatureChainParserMixin,
    FeatureGroup,
    FeatureSet,
    collect_match_rejections,
    is_match_abort,
)
from mloda.user import DataType, Feature, Options

from mloda.community.feature_groups.data_operations.base import KeyColumnInputsMixin, always_required
from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.base import (
    _AGGREGATION_TYPES,
    FrameAggregateFeatureGroup,
)
from mloda.testing.feature_groups.data_operations.helpers import load_implementation, make_feature_set
from mloda.testing.feature_groups.data_operations.match_validation import MatchValidationTestBase, TokenCase
from mloda.testing.feature_groups.data_operations.operation_group import ImplSpec, OperationGroupTestBase
from mloda.testing.feature_groups.data_operations.row_preserving.frame_aggregate.frame_aggregate import (
    config_frame_options,
    time_frame_options,
)


class TestPatternParsing:
    """Tests for _parse_frame_feature pattern matching."""

    @pytest.mark.parametrize(
        ("feature_name", "expected"),
        [
            ("sales__sum_rolling_3", {"aggregation_type": "sum", "frame_size": 3}),
            ("sales__avg_7_day_window", {"aggregation_type": "avg", "frame_size": 7, "frame_unit": "day"}),
            ("sales__cumsum", {"aggregation_type": "sum"}),
            ("sales__expanding_avg", {"aggregation_type": "avg"}),
        ],
    )
    def test_resolve_feature_name_bindings(self, feature_name: str, expected: dict[str, Any]) -> None:
        assert FrameAggregateFeatureGroup.resolve_feature_name(feature_name).bindings == expected

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

    @pytest.mark.parametrize(
        "feature_name",
        [
            pytest.param("plain_feature", id="plain_feature"),
            pytest.param("sales__sum_rolling_3\n", id="rolling_trailing_newline"),
            pytest.param("price__avg_7_day_window\n", id="time_window_trailing_newline"),
            pytest.param("sales__cumsum\n", id="cumulative_trailing_newline"),
            pytest.param("score__expanding_avg\n", id="expanding_trailing_newline"),
        ],
    )
    def test_no_match(self, feature_name: str) -> None:
        result = FrameAggregateFeatureGroup._parse_frame_feature(feature_name)
        assert result is None

    @pytest.mark.parametrize(
        ("feature_name", "expected_fields"),
        [
            pytest.param("value__max_rolling_100", {"frame_size": 100, "agg_type": "max"}, id="rolling_large_window"),
            pytest.param("temp__min_24_hour_window", {"frame_unit": "hour", "frame_size": 24}, id="time_window_hour"),
            pytest.param("price__cummin", {"agg_type": "min", "frame_type": "cumulative"}, id="cummin"),
            pytest.param("price__cummax", {"agg_type": "max", "frame_type": "cumulative"}, id="cummax"),
            pytest.param("price__cumcount", {"agg_type": "count", "frame_type": "cumulative"}, id="cumcount"),
        ],
    )
    def test_parse_frame_fields(self, feature_name: str, expected_fields: dict[str, Any]) -> None:
        result = FrameAggregateFeatureGroup._parse_frame_feature(feature_name)
        assert result is not None
        for field, expected in expected_fields.items():
            assert result[field] == expected, f"{feature_name}: {field}"

    def test_multi_source_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="allows at most 1 in_feature"):
            FrameAggregateFeatureGroup._extract_single_source_feature(Feature("a&b__sum_rolling_3"))


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

    @pytest.mark.parametrize(
        ("feature_name", "expected"),
        [
            pytest.param("sales__sum_rolling_3", True, id="rolling_string"),
            pytest.param("sales__avg_7_day_window", True, id="time_window_string"),
            pytest.param("sales__cumsum", True, id="cumulative_string"),
            pytest.param("sales__expanding_avg", True, id="expanding_string"),
            # cumavg is a valid cumulative operation (cumulative and expanding are aliases).
            pytest.param("sales__cumavg", True, id="cumavg"),
            pytest.param("sales__unknown_rolling_3", False, id="invalid_agg_type"),
            pytest.param("sales__avg_7_banana_window", False, id="invalid_time_unit"),
            pytest.param("plain_feature", False, id="plain_feature_without_config"),
            pytest.param("sales__sum_rolling_3\n", False, id="trailing_newline"),
        ],
    )
    def test_match_by_name(self, feature_name: str, expected: bool) -> None:
        options = self._base_options()
        result = FrameAggregateFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is expected

    def test_rejects_no_partition_by(self) -> None:
        options = Options(context={"order_by": "timestamp"})
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("sales__sum_rolling_3", options, None)
        assert result is False

    def test_rejects_no_order_by(self) -> None:
        options = Options(context={"partition_by": ["region"]})
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("sales__sum_rolling_3", options, None)
        assert result is False

    @pytest.mark.parametrize(
        "feature_name",
        [
            "sales__sum_rolling_3",
            "sales__avg_7_day_window",
            "sales__cumsum",
            "sales__expanding_avg",
        ],
    )
    def test_rejects_no_order_by_every_shape(self, feature_name: str) -> None:
        """order_by is required on every name shape, unsized cumulative and expanding frames included."""
        options = Options(context={"partition_by": ["region"]})
        result = FrameAggregateFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is False, f"Name path should require order_by: {feature_name}"

    def test_rejects_wrong_typed_order_by(self) -> None:
        """order_by must be a column reference, not any scalar."""
        options = Options(context={"partition_by": ["region"], "order_by": 123})
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("sales__sum_rolling_3", options, None)
        assert result is False

    def test_rejects_partition_by_as_string(self) -> None:
        options = Options(context={"partition_by": "region", "order_by": "timestamp"})
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("sales__sum_rolling_3", options, None)
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

    def test_config_rejects_missing_order_by(self) -> None:
        options = Options(
            context={
                "aggregation_type": "sum",
                "frame_type": "rolling",
                "frame_size": 3,
                "in_features": "sales",
                "partition_by": ["region"],
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

    def test_frame_unit_guard_rejects_multi_element_value(self) -> None:
        # The key's own guard, independent of the time-unit table.
        guard = FrameAggregateFeatureGroup.PROPERTY_MAPPING[FrameAggregateFeatureGroup.FRAME_UNIT].match_guard
        assert guard is not None
        assert guard(("day", "week")) is False
        assert guard(("day",)) is True
        assert guard("day") is True

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

    @pytest.mark.parametrize(
        ("overrides", "key", "frame_type"),
        [
            ({"frame_type": "cumulative", "frame_size": 3}, "frame_size", "cumulative"),
            ({"frame_type": "expanding", "frame_unit": "day"}, "frame_unit", "expanding"),
            ({"frame_type": "rolling", "frame_size": 3, "frame_unit": "day"}, "frame_unit", "rolling"),
        ],
        ids=["cumulative_size", "expanding_unit", "rolling_unit"],
    )
    def test_own_unused_frame_option_raises(self, overrides: dict[str, Any], key: str, frame_type: str) -> None:
        with pytest.raises(ValueError, match=rf"option '{key}'.*'{frame_type}'.*remove") as exc_info:
            FrameAggregateFeatureGroup.match_feature_group_criteria("my_result", self._options(**overrides), None)
        assert is_match_abort(exc_info.value)

    def test_inherited_frame_type_does_not_trigger_the_unused_option_raise(self) -> None:
        options = self._options(frame_size=3)
        options.inherit_from(Options(group={"frame_type": "cumulative"}))
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False


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

    @pytest.mark.parametrize(
        ("feature_name", "frame_type"),
        [
            pytest.param("sales__sum_rolling_3", "rolling", id="rolling_name"),
            pytest.param("sales__avg_7_day_window", "time", id="time_window_name"),
        ],
    )
    def test_name_with_frame_type_option(self, feature_name: str, frame_type: str) -> None:
        options = Options(context={"frame_type": frame_type, "partition_by": ["region"], "order_by": "timestamp"})
        result = FrameAggregateFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is True

    def test_rolling_name_with_propagated_frame_type(self) -> None:
        """A propagated frame_type must behave like a directly set one."""
        consumer = Options(context={"frame_type": "rolling"}, propagate_context_keys=frozenset({"frame_type"}))
        options = Options(context={"partition_by": ["region"], "order_by": "timestamp"})
        options.inherit_from(consumer)
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("sales__sum_rolling_3", options, None)
        assert result is True


class TestNameSuppliedFrameTypeRequiredWhen:
    """The name supplies frame_type just like frame_size and frame_unit, so a required_when on it
    should not fire on the name path; the one test below documents that this guarantee does not yet
    hold under mloda 0.11.0 (hence its xfail), while the config path, which has no name to supply it,
    still enforces it.
    """

    @staticmethod
    def _frame_type_required() -> type[FrameAggregateFeatureGroup]:
        class _FrameTypeRequired(FrameAggregateFeatureGroup):
            PROPERTY_MAPPING = {
                **FrameAggregateFeatureGroup.PROPERTY_MAPPING,
                FrameAggregateFeatureGroup.FRAME_TYPE: dataclasses.replace(
                    FrameAggregateFeatureGroup.PROPERTY_MAPPING[FrameAggregateFeatureGroup.FRAME_TYPE],
                    required_when=always_required,
                ),
            }

        return _FrameTypeRequired

    @pytest.mark.xfail(
        reason=(
            "mloda 0.11.0's required_when guard reads name patterns from PREFIX_PATTERN/SUFFIX_PATTERN "
            "only, not an overridden _get_prefix_patterns(), and frame_aggregate exposes only the "
            "rolling shape as PREFIX_PATTERN with none of its four name patterns capturing frame_type "
            "as a group. So the guard never learns frame_type from the name and treats an always-"
            "required frame_type as absent. Production PROPERTY_MAPPING sidesteps this by never "
            "declaring required_when on FRAME_TYPE/FRAME_SIZE/FRAME_UNIT; this synthetic subclass exists "
            "solely to document the guard's blind spot."
        ),
        strict=True,
    )
    def test_name_path_not_rejected_by_frame_type_required_when(self) -> None:
        """A rolling name carries its frame type, so an always-on frame_type requirement must not reject it."""
        options = Options(context={"partition_by": ["region"], "order_by": "timestamp"})
        result = self._frame_type_required().match_feature_group_criteria("sales__sum_rolling_3", options, None)
        assert result is True

    def test_config_path_still_enforces_frame_type(self) -> None:
        """Control: a config feature without frame_type stays a non-match."""
        options = Options(
            context={
                "aggregation_type": "sum",
                "in_features": "sales",
                "partition_by": ["region"],
                "order_by": "timestamp",
            }
        )
        result = self._frame_type_required().match_feature_group_criteria("my_result", options, None)
        assert result is False


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


class TestFrameNameAgreement:
    """A frame option (forwarded or own) that contradicts the name must raise, never be silently overridden."""

    def test_name_agreement_runs_before_match_guards(self) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "timestamp", "frame_size": 0})
        with pytest.raises(ValueError, match=r"option 'frame_size' is int 0, but the feature name encodes '3'"):
            FrameAggregateFeatureGroup.match_feature_group_criteria("sales__sum_rolling_3", options, None)

    def test_forwarded_aggregation_type_mismatch_raises(self) -> None:
        consumer = Options(group={"aggregation_type": "max"})
        options = Options(context={"partition_by": ["region"], "order_by": "timestamp"})
        options.inherit_from(consumer)
        with pytest.raises(ValueError):
            FrameAggregateFeatureGroup.match_feature_group_criteria("sales__sum_rolling_3", options, None)

    @pytest.mark.parametrize("path", ["group", "context"])
    def test_forwarded_frame_type_mismatch_raises_via_path(self, path: str) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "timestamp"})
        if path == "group":
            options.inherit_from(Options(group={"frame_type": "time"}))
        else:
            options.inherit_from(
                Options(context={"frame_type": "time"}), inherit_context_keys=frozenset({"frame_type"})
            )
        with pytest.raises(ValueError, match="frame_type") as exc_info:
            FrameAggregateFeatureGroup.match_feature_group_criteria("sales__sum_rolling_3", options, None)
        assert "was forwarded from a consumer" in str(exc_info.value)

    @pytest.mark.parametrize("path", ["group", "context"])
    def test_forwarded_frame_size_mismatch_raises_via_path(self, path: str) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "timestamp"})
        if path == "group":
            options.inherit_from(Options(group={"frame_size": 99}))
        else:
            options.inherit_from(Options(context={"frame_size": 99}), inherit_context_keys=frozenset({"frame_size"}))
        with pytest.raises(ValueError, match="frame_size") as exc_info:
            FrameAggregateFeatureGroup.match_feature_group_criteria("sales__sum_rolling_3", options, None)
        assert "was forwarded from a consumer" in str(exc_info.value)

    @pytest.mark.parametrize("path", ["group", "context"])
    def test_forwarded_frame_unit_mismatch_raises_via_path(self, path: str) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "timestamp"})
        if path == "group":
            options.inherit_from(Options(group={"frame_unit": "hour"}))
        else:
            options.inherit_from(
                Options(context={"frame_unit": "hour"}), inherit_context_keys=frozenset({"frame_unit"})
            )
        with pytest.raises(ValueError, match="frame_unit") as exc_info:
            FrameAggregateFeatureGroup.match_feature_group_criteria("sales__avg_7_day_window", options, None)
        assert "was forwarded from a consumer" in str(exc_info.value)

    @pytest.mark.parametrize(
        ("feature_name", "key", "value", "binding", "type_name"),
        [
            ("sales__sum_rolling_3", "frame_size", 5, "3", "int"),
            ("sales__sum_rolling_3", "frame_type", "time", "rolling", "str"),
            ("sales__avg_7_day_window", "frame_size", 5, "7", "int"),
            ("sales__avg_7_day_window", "frame_unit", "hour", "day", "str"),
            ("sales__avg_7_day_window", "frame_type", "rolling", "time", "str"),
            ("sales__cumsum", "frame_type", "rolling", "cumulative", "str"),
        ],
    )
    @pytest.mark.parametrize("placement", ["group", "context"])
    def test_own_contradictory_frame_param_raises(
        self, placement: str, feature_name: str, key: str, value: Any, binding: str, type_name: str
    ) -> None:
        base = {"partition_by": ["region"], "order_by": "timestamp"}
        if placement == "group":
            options = Options(group={key: value}, context=base)
        else:
            options = Options(context={key: value, **base})
        with pytest.raises(
            ValueError,
            match=rf"option '{key}' is {type_name} {re.escape(repr(value))}, but the feature name encodes '{binding}'\. The name is authoritative",
        ):
            FrameAggregateFeatureGroup.match_feature_group_criteria(feature_name, options, None)

    @pytest.mark.parametrize("wrap", [False, True], ids=["scalar", "singleton_tuple"])
    @pytest.mark.parametrize("placement", ["group", "context"])
    def test_own_matching_frame_params_are_accepted(self, placement: str, wrap: bool) -> None:
        values: dict[str, Any] = {"frame_type": "time", "frame_size": 7, "frame_unit": "day"}
        if wrap:
            values = {k: (v,) for k, v in values.items()}
        base = {"partition_by": ["region"], "order_by": "timestamp"}
        if placement == "group":
            options = Options(group=values, context=base)
        else:
            options = Options(context={**values, **base})
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("sales__avg_7_day_window", options, None)
        assert result is True

    @pytest.mark.parametrize(
        ("origin", "placement"),
        [("forwarded", "group"), ("own", "group"), ("own", "context")],
    )
    def test_frame_type_mismatch_env_downgrade_is_accepted(
        self, origin: str, placement: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("MLODA_ALLOW_FORWARDED_NAME_MISMATCH", "1")
        base = {"partition_by": ["region"], "order_by": "timestamp"}
        if origin == "forwarded":
            options = Options(context=base)
            options.inherit_from(Options(group={"frame_type": "time"}))
        elif placement == "group":
            options = Options(group={"frame_type": "time"}, context=base)
        else:
            options = Options(context={"frame_type": "time", **base})
        with caplog.at_level(logging.WARNING):
            result = FrameAggregateFeatureGroup.match_feature_group_criteria("sales__sum_rolling_3", options, None)
        assert result is True
        assert any(r.levelno == logging.WARNING and "frame_type" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize("wrap", [False, True], ids=["scalar", "singleton_tuple"])
    @pytest.mark.parametrize("path", ["group", "context"])
    def test_matching_forwarded_frame_params_is_accepted(self, path: str, wrap: bool) -> None:
        values: dict[str, Any] = {"frame_type": "time", "frame_size": 7, "frame_unit": "day"}
        if wrap:
            values = {k: (v,) for k, v in values.items()}
        options = Options(context={"partition_by": ["region"], "order_by": "timestamp"})
        if path == "group":
            options.inherit_from(Options(group=values))
        else:
            options.inherit_from(Options(context=values), inherit_context_keys=frozenset(values))
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("sales__avg_7_day_window", options, None)
        assert result is True

    @pytest.mark.parametrize("feature_name", ["sales__cumsum", "sales__expanding_avg"])
    def test_sizeless_shapes_ignore_forwarded_frame_size_and_unit(self, feature_name: str) -> None:
        consumer = Options(group={"frame_size": 99, "frame_unit": "hour"})
        options = Options(context={"partition_by": ["region"], "order_by": "timestamp"})
        options.inherit_from(consumer)
        result = FrameAggregateFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is True

    def test_rolling_ignores_forwarded_frame_unit(self) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "timestamp"})
        options.inherit_from(Options(group={"frame_unit": "hour"}))
        result = FrameAggregateFeatureGroup.match_feature_group_criteria("sales__sum_rolling_3", options, None)
        assert result is True

    @pytest.mark.parametrize(
        ("feature_name", "key", "value", "frame_type"),
        [
            ("sales__cumsum", "frame_size", 3, "cumulative"),
            ("sales__expanding_avg", "frame_size", 3, "expanding"),
            ("sales__cumsum", "frame_unit", "day", "cumulative"),
            ("sales__expanding_avg", "frame_unit", "day", "expanding"),
            ("sales__sum_rolling_3", "frame_unit", "day", "rolling"),
        ],
    )
    @pytest.mark.parametrize("placement", ["group", "context"])
    def test_own_unused_frame_option_raises(
        self, placement: str, feature_name: str, key: str, value: Any, frame_type: str
    ) -> None:
        base = {"partition_by": ["region"], "order_by": "timestamp"}
        if placement == "group":
            options = Options(group={key: value}, context=base)
        else:
            options = Options(context={key: value, **base})
        with pytest.raises(ValueError, match=rf"option '{key}'.*'{frame_type}'.*remove") as exc_info:
            FrameAggregateFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert feature_name in str(exc_info.value)
        assert is_match_abort(exc_info.value)

    @pytest.mark.parametrize(
        ("feature_name", "frame_type"),
        [("sales__cumsum", "expanding"), ("sales__expanding_avg", "cumulative")],
    )
    @pytest.mark.parametrize("wrap", [False, True], ids=["scalar", "singleton_tuple"])
    @pytest.mark.parametrize("origin", ["own", "forwarded"])
    def test_expanding_and_cumulative_frame_types_agree(
        self, origin: str, wrap: bool, feature_name: str, frame_type: str
    ) -> None:
        value: Any = (frame_type,) if wrap else frame_type
        base = {"partition_by": ["region"], "order_by": "timestamp"}
        if origin == "own":
            options = Options(group={"frame_type": value}, context=base)
        else:
            options = Options(context=base)
            options.inherit_from(Options(group={"frame_type": value}))
        result = FrameAggregateFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is True

    def test_contradiction_is_reported_before_the_unused_option(self) -> None:
        options = Options(
            context={"frame_type": "rolling", "frame_size": 3, "partition_by": ["region"], "order_by": "timestamp"}
        )
        with pytest.raises(ValueError, match=r"option 'frame_type' is .* but the feature name encodes 'cumulative'"):
            FrameAggregateFeatureGroup.match_feature_group_criteria("sales__cumsum", options, None)


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

    def test_unknown_aggregation_name_records_the_rejected_binding(self) -> None:
        with collect_match_rejections() as reasons:
            assert (
                FrameAggregateFeatureGroup.match_feature_group_criteria(
                    "sales__banana_rolling_3", self._pattern_options(), None
                )
                is False
            )
        reason = reasons[FrameAggregateFeatureGroup.get_class_name()].reason
        assert "'banana'" in reason
        assert "aggregation_type" in reason


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

    def test_extract_from_name_without_order_by_raises(self) -> None:
        """The order_by backstop sits above the name/config branch, so the name path hits it too."""
        with pytest.raises(ValueError, match="order_by"):
            FrameAggregateFeatureGroup._extract_params(
                Feature("sales__sum_rolling_3", options=Options(context={"partition_by": ["region"]}))
            )


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
    def malformed_operations(cls) -> set[str]:
        return {"banana", "mode", "nunique", "first"}

    @classmethod
    def token_cases(cls) -> list[TokenCase]:
        # frame_type and frame_unit are operation tokens of their own, and frame_type is what
        # _needs_frame_size / _needs_frame_unit read to decide which of the two a state requires.
        return [
            *super().token_cases(),
            TokenCase("frame_type", "rolling", "time"),
            TokenCase("frame_unit", "day", "week", context={"frame_type": "time"}),
            # Sized frames require frame_size, unsized ones do not.
            TokenCase("frame_type", "rolling", without=("frame_size",), matches=False),
            TokenCase("frame_type", "time", without=("frame_size",), matches=False),
            TokenCase("frame_type", "cumulative", without=("frame_size",)),
            TokenCase("frame_type", "expanding", without=("frame_size",)),
            # Only a time frame requires frame_unit, which additional_match_options() does not carry.
            TokenCase("frame_type", "time", matches=False),
            # order_by and frame_size are scalar too: one column, one positive int, so a
            # zero-sized frame and a bool stay out at every arity.
            TokenCase("order_by", "timestamp", "region", required=True),
            TokenCase("frame_size", 3, 5, invalid=(0, True)),
        ]

    @classmethod
    def dispatch_values(cls, options: Options) -> list[Any]:
        params = FrameAggregateFeatureGroup._extract_params(Feature("my_result", options=options))
        return [
            params["agg_type"],
            params["frame_type"],
            params["frame_unit"],
            params["order_by"],
            params["frame_size"],
        ]


_PACKAGE = "mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate"
_ROLLING_OPTIONS = Options(context={"partition_by": ["region"], "order_by": "ts"})


_ENGINE = "mloda_plugins.compute_framework.base_implementations"


class TestFrameAggregateOperationGroup(OperationGroupTestBase):
    @classmethod
    def group_class(cls) -> Any:
        return FrameAggregateFeatureGroup

    @classmethod
    def backend_package(cls) -> str:
        return "mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate"

    @classmethod
    def implementations(cls) -> list[ImplSpec]:
        return [
            ImplSpec(
                "pandas",
                "pandas_frame_aggregate",
                "PandasFrameAggregate",
                f"{_ENGINE}.pandas.dataframe",
                "PandasDataFrame",
                divergent=frozenset({"sum", "avg", "count", "std", "var"}),
                dependency="pandas",
            ),
            ImplSpec(
                "polars_lazy",
                "polars_lazy_frame_aggregate",
                "PolarsLazyFrameAggregate",
                f"{_ENGINE}.polars.lazy_dataframe",
                "PolarsLazyDataFrame",
                dependency="polars",
            ),
            ImplSpec(
                "duckdb",
                "duckdb_frame_aggregate",
                "DuckdbFrameAggregate",
                f"{_ENGINE}.duckdb.duckdb_framework",
                "DuckDBFramework",
                dependency="duckdb",
            ),
            ImplSpec(
                "sqlite",
                "sqlite_frame_aggregate",
                "SqliteFrameAggregate",
                f"{_ENGINE}.sqlite.sqlite_framework",
                "SqliteFramework",
                supported=frozenset({"sum", "avg", "count", "min", "max"}),
                divergent=frozenset({"sum", "avg", "count"}),
            ),
            ImplSpec(
                "python_dict",
                "python_dict_frame_aggregate",
                "PythonDictFrameAggregate",
                f"{_ENGINE}.python_dict.python_dict_framework",
                "PythonDictFramework",
            ),
        ]

    @classmethod
    def subtype_key(cls) -> str | None:
        return None

    @classmethod
    def loop_method(cls) -> str:
        return "aggregate"

    @classmethod
    def compute_method(cls) -> str:
        return "_compute_frame"

    @classmethod
    def probe_feature_set(cls) -> FeatureSet:
        return make_feature_set("value_int__sum_rolling_3", ["region"], "value_int")

    @classmethod
    def name_probes(cls) -> list[tuple[str, str]]:
        return [
            ("x__sum_rolling_3", "sum"),
            ("x__avg_7_day_window", "avg"),
            ("x__cumsum", "sum"),
            ("x__expanding_max", "max"),
        ]

    @classmethod
    def option_probes(cls) -> list[tuple[str, Options, str]]:
        return [("my_result", config_frame_options("median", "rolling"), "median")]

    @classmethod
    def unresolved_probes(cls) -> list[tuple[str, Options]]:
        return [("plain_feature", Options())]

    @classmethod
    def universe(cls) -> frozenset[str]:
        return frozenset({"sum", "avg", "count", "min", "max", "std", "var", "median"})

    def test_group_keeps_both_mixins_ahead_of_the_feature_group(self) -> None:
        # OperationFeatureGroup precedes the mixins here, so mixins() does not apply.
        mro = FrameAggregateFeatureGroup.__mro__
        assert mro.index(KeyColumnInputsMixin) < mro.index(FeatureGroup)
        assert mro.index(FeatureChainParserMixin) < mro.index(FeatureGroup)


class TestTimeUnits:
    """Time units are declared per Implementation; supported and divergent pins live in TestFrameAggregateOperationGroup."""

    @pytest.mark.parametrize(
        ("module", "name"),
        [
            ("pandas_frame_aggregate", "PandasFrameAggregate"),
            ("sqlite_frame_aggregate", "SqliteFrameAggregate"),
        ],
    )
    def test_time_units_live_on_the_implementation(self, module: str, name: str) -> None:
        units = load_implementation(_PACKAGE, module, name).SUPPORTED_TIME_UNITS
        assert set(units) == {"second", "minute", "hour", "day", "week"}


class TestImplementationAccepts:
    """Per-framework narrowing of frame features is the Implementation's ``accepts``."""

    @pytest.mark.parametrize(
        ("module", "name"),
        [
            ("pandas_frame_aggregate", "PandasFrameAggregate"),
            ("sqlite_frame_aggregate", "SqliteFrameAggregate"),
        ],
    )
    def test_month_and_year_windows_are_rejected(self, module: str, name: str) -> None:
        impl = load_implementation(_PACKAGE, module, name)
        assert not impl.accepts("value__sum_1_month_window", _ROLLING_OPTIONS)
        assert not impl.accepts("value__sum_1_year_window", _ROLLING_OPTIONS)
        assert not impl.accepts("value_time_frame", time_frame_options("month"))
        assert not impl.accepts("value_time_frame", time_frame_options("year"))

    @pytest.mark.parametrize(
        ("module", "name"),
        [
            ("pandas_frame_aggregate", "PandasFrameAggregate"),
            ("sqlite_frame_aggregate", "SqliteFrameAggregate"),
        ],
    )
    def test_day_and_week_windows_are_accepted(self, module: str, name: str) -> None:
        impl = load_implementation(_PACKAGE, module, name)
        assert impl.accepts("value__sum_3_day_window", _ROLLING_OPTIONS)
        assert impl.accepts("value__sum_1_week_window", _ROLLING_OPTIONS)
        assert impl.accepts("value_time_frame", time_frame_options("day"))
        assert impl.accepts("value_time_frame", time_frame_options("week"))

    @pytest.mark.parametrize("agg_type", ["std", "var", "median"])
    @pytest.mark.parametrize("frame_type", ["cumulative", "expanding"])
    def test_polars_rejects_std_var_median_on_cumulative_and_expanding(self, agg_type: str, frame_type: str) -> None:
        impl = load_implementation(_PACKAGE, "polars_lazy_frame_aggregate", "PolarsLazyFrameAggregate")
        assert not impl.accepts("value_frame", config_frame_options(agg_type, frame_type))

    @pytest.mark.parametrize(
        "feature_name", ["value__cumstd", "value__cumvar", "value__cummedian", "value__expanding_std"]
    )
    def test_polars_rejects_cumulative_names(self, feature_name: str) -> None:
        impl = load_implementation(_PACKAGE, "polars_lazy_frame_aggregate", "PolarsLazyFrameAggregate")
        assert not impl.accepts(feature_name, _ROLLING_OPTIONS)

    @pytest.mark.parametrize("agg_type", ["std", "var", "median"])
    def test_polars_accepts_std_var_median_on_rolling_and_time(self, agg_type: str) -> None:
        impl = load_implementation(_PACKAGE, "polars_lazy_frame_aggregate", "PolarsLazyFrameAggregate")
        assert impl.accepts("value_frame", config_frame_options(agg_type, "rolling"))
        assert impl.accepts(f"value__{agg_type}_rolling_3", _ROLLING_OPTIONS)
        assert impl.accepts(f"value__{agg_type}_7_day_window", _ROLLING_OPTIONS)

    @pytest.mark.parametrize(
        ("module", "name", "accepted", "rejected"),
        [
            ("pandas_frame_aggregate", "PandasFrameAggregate", "value__sum_3_day_window", "value__sum_1_month_window"),
            ("sqlite_frame_aggregate", "SqliteFrameAggregate", "value__sum_3_day_window", "value__sum_1_month_window"),
            ("polars_lazy_frame_aggregate", "PolarsLazyFrameAggregate", "value__std_rolling_3", "value__cumstd"),
        ],
    )
    @pytest.mark.parametrize("part", [None, "sum"])
    def test_accepts_takes_the_part_keyword_without_changing_the_result(
        self, module: str, name: str, accepted: str, rejected: str, part: str | None
    ) -> None:
        impl = load_implementation(_PACKAGE, module, name)
        assert impl.accepts(accepted, _ROLLING_OPTIONS, part=part) == impl.accepts(accepted, _ROLLING_OPTIONS)
        assert impl.accepts(rejected, _ROLLING_OPTIONS, part=part) == impl.accepts(rejected, _ROLLING_OPTIONS)
        assert impl.accepts(accepted, _ROLLING_OPTIONS, part=part)
        assert not impl.accepts(rejected, _ROLLING_OPTIONS, part=part)

    @pytest.mark.parametrize(
        ("module", "name"),
        [
            ("duckdb_frame_aggregate", "DuckdbFrameAggregate"),
            ("python_dict_frame_aggregate", "PythonDictFrameAggregate"),
        ],
    )
    def test_unrestricted_implementations_accept_everything(self, module: str, name: str) -> None:
        impl = load_implementation(_PACKAGE, module, name)
        assert impl.accepts("value__sum_1_month_window", _ROLLING_OPTIONS)
        assert impl.accepts("value__cumstd", _ROLLING_OPTIONS)
        assert impl.accepts("value_time_frame", time_frame_options("year"))
        assert impl.accepts("value_frame", config_frame_options("median", "cumulative"))
