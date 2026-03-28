"""Tests for OffsetFeatureGroup base class."""

from __future__ import annotations

import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.community.feature_groups.data_operations.row_preserving.offset.base import OffsetFeatureGroup


class TestClassAttributes:
    def test_prefix_pattern_exists(self) -> None:
        assert hasattr(OffsetFeatureGroup, "PREFIX_PATTERN")

    def test_offset_types_exists(self) -> None:
        assert hasattr(OffsetFeatureGroup, "OFFSET_TYPES")
        assert "first_value" in OffsetFeatureGroup.OFFSET_TYPES
        assert "last_value" in OffsetFeatureGroup.OFFSET_TYPES

    def test_supports_lag(self) -> None:
        assert OffsetFeatureGroup._supports_offset_type("lag_1")
        assert OffsetFeatureGroup._supports_offset_type("lag_5")

    def test_supports_lead(self) -> None:
        assert OffsetFeatureGroup._supports_offset_type("lead_1")

    def test_supports_diff(self) -> None:
        assert OffsetFeatureGroup._supports_offset_type("diff_1")

    def test_supports_pct_change(self) -> None:
        assert OffsetFeatureGroup._supports_offset_type("pct_change_1")

    def test_supports_first_value(self) -> None:
        assert OffsetFeatureGroup._supports_offset_type("first_value")

    def test_supports_last_value(self) -> None:
        assert OffsetFeatureGroup._supports_offset_type("last_value")

    def test_rejects_invalid(self) -> None:
        assert not OffsetFeatureGroup._supports_offset_type("lag_0")
        assert not OffsetFeatureGroup._supports_offset_type("unknown")

    def test_rejects_non_numeric_suffix(self) -> None:
        assert not OffsetFeatureGroup._supports_offset_type("lag_abc")
        assert not OffsetFeatureGroup._supports_offset_type("lead_")

    def test_rejects_empty_string(self) -> None:
        assert not OffsetFeatureGroup._supports_offset_type("")

    def test_min_max_in_features(self) -> None:
        assert OffsetFeatureGroup.MIN_IN_FEATURES == 1
        assert OffsetFeatureGroup.MAX_IN_FEATURES == 1


class TestPatternMatching:
    @pytest.mark.parametrize(
        "feature_name",
        [
            "value_int__lag_1_offset",
            "value_int__lead_1_offset",
            "value_int__diff_1_offset",
            "value_int__pct_change_1_offset",
            "value_int__first_value_offset",
            "value_int__last_value_offset",
        ],
    )
    def test_matches_offset_types(self, feature_name: str) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "value_int"})
        result = OffsetFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is True

    def test_no_match_wrong_suffix(self) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "value_int"})
        assert not OffsetFeatureGroup.match_feature_group_criteria("value_int__lag_1_ranked", options, None)

    def test_no_match_invalid_type(self) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "value_int"})
        assert not OffsetFeatureGroup.match_feature_group_criteria("value_int__unknown_offset", options, None)


class TestPatternParsing:
    def test_parse_lag(self) -> None:
        assert OffsetFeatureGroup.get_offset_type("value_int__lag_1_offset") == "lag_1"

    def test_parse_lead(self) -> None:
        assert OffsetFeatureGroup.get_offset_type("value_int__lead_3_offset") == "lead_3"

    def test_parse_first_value(self) -> None:
        assert OffsetFeatureGroup.get_offset_type("value_int__first_value_offset") == "first_value"

    def test_parse_source_feature(self) -> None:
        from mloda.user import Feature

        feature = Feature(
            "value_int__lag_1_offset",
            options=Options(context={"partition_by": ["region"], "order_by": "value_int"}),
        )
        assert OffsetFeatureGroup._extract_source_features(feature) == ["value_int"]


class TestConfigValidation:
    def test_partition_by_required(self) -> None:
        options = Options(context={"order_by": "value_int"})
        assert not OffsetFeatureGroup.match_feature_group_criteria("value_int__lag_1_offset", options, None)

    def test_order_by_required(self) -> None:
        options = Options(context={"partition_by": ["region"]})
        assert not OffsetFeatureGroup.match_feature_group_criteria("value_int__lag_1_offset", options, None)

    def test_valid_config(self) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "value_int"})
        assert OffsetFeatureGroup.match_feature_group_criteria("value_int__lag_1_offset", options, None)

    def test_partition_by_as_tuple(self) -> None:
        """partition_by should accept tuples (mloda converts lists to tuples internally)."""
        options = Options(context={"partition_by": ("region",), "order_by": "value_int"})
        assert OffsetFeatureGroup.match_feature_group_criteria("value_int__lag_1_offset", options, None)

    def test_partition_by_rejects_string(self) -> None:
        options = Options(context={"partition_by": "region", "order_by": "value_int"})
        assert not OffsetFeatureGroup.match_feature_group_criteria("value_int__lag_1_offset", options, None)

    def test_partition_by_rejects_non_string_items(self) -> None:
        options = Options(context={"partition_by": [123], "order_by": "value_int"})
        assert not OffsetFeatureGroup.match_feature_group_criteria("value_int__lag_1_offset", options, None)

    def test_order_by_rejects_non_string(self) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": 123})
        assert not OffsetFeatureGroup.match_feature_group_criteria("value_int__lag_1_offset", options, None)


class TestConfigBasedFeatures:
    def test_config_based_match(self) -> None:
        options = Options(
            context={
                "offset_type": "lag_1",
                "in_features": "value_int",
                "partition_by": ["region"],
                "order_by": "value_int",
            }
        )
        assert OffsetFeatureGroup.match_feature_group_criteria("my_result", options, None)

    def test_config_based_calculate_feature(self) -> None:
        import pyarrow as pa

        from mloda.core.abstract_plugins.components.feature_set import FeatureSet
        from mloda.community.feature_groups.data_operations.row_preserving.offset.pyarrow_offset import PyArrowOffset
        from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
        from mloda.user import Feature

        table = PyArrowDataOpsTestDataCreator.create()
        feature = Feature(
            "my_lag",
            options=Options(
                context={
                    "offset_type": "lag_1",
                    "in_features": "value_int",
                    "partition_by": ["region"],
                    "order_by": "value_int",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = PyArrowOffset.calculate_feature(table, fs)
        assert isinstance(result, pa.Table)
        assert "my_lag" in result.column_names
        assert result.num_rows == 12
