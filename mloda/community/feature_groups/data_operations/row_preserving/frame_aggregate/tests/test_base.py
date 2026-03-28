"""Tests for FrameAggregateFeatureGroup base class."""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.base import (
    FrameAggregateFeatureGroup,
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
