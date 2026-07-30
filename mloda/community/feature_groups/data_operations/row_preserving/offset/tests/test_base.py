"""Tests for OffsetFeatureGroup base class."""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.match_validation import MatchValidationTestBase
from mloda.user import DataType, Feature

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
        from mloda.testing.feature_groups.data_operations.row_preserving.offset.reference import ReferenceOffset
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

        result = ReferenceOffset.calculate_feature(table, fs)
        assert isinstance(result, pa.Table)
        assert "my_lag" in result.column_names
        assert result.num_rows == 12


class TestConfigBasedOffsetTypeValidation:
    """The config path must reject exactly the offset types the feature-name path rejects.

    Matching may not defer offset-type validation to compute time: an unsupported
    offset_type has to be a non-match at discovery.
    """

    def _options(self, offset_type: Any) -> Options:
        return Options(
            context={
                "offset_type": offset_type,
                "in_features": "value_int",
                "partition_by": ["region"],
                "order_by": "value_int",
            }
        )

    @pytest.mark.parametrize(
        "offset_type",
        ["lag_1", "lead_2", "diff_1", "pct_change_3", "first_value", "last_value"],
    )
    def test_config_based_match_accepts_supported_offset_types(self, offset_type: str) -> None:
        result = OffsetFeatureGroup.match_feature_group_criteria("my_result", self._options(offset_type), None)
        assert result is True, f"Config path should accept: {offset_type}"

    @pytest.mark.parametrize("offset_type", ["banana", "lag_0", "lag_-1", "lag_abc", "lead_", ""])
    def test_config_based_match_rejects_unsupported_offset_types(self, offset_type: str) -> None:
        result = OffsetFeatureGroup.match_feature_group_criteria("my_result", self._options(offset_type), None)
        assert result is False, f"Config path should reject: {offset_type!r}"


class TestSingleTokenContainers:
    """A single-element container holds exactly one offset token, so it must reach dispatch unwrapped."""

    def _options(self, offset_type: Any) -> Options:
        return Options(
            context={
                "offset_type": offset_type,
                "in_features": "value_int",
                "partition_by": ["region"],
                "order_by": "value_int",
            }
        )

    @pytest.mark.parametrize("offset_type", [("lag_1",), ["lag_1"]])
    def test_single_element_offset_type(self, offset_type: Any) -> None:
        result = OffsetFeatureGroup.match_feature_group_criteria("my_result", self._options(offset_type), None)
        assert result is True, f"Config path should accept: {offset_type!r}"
        feature = Feature("my_result", options=self._options(offset_type))
        assert OffsetFeatureGroup._extract_offset_type(feature) == "lag_1"

    @pytest.mark.parametrize("offset_type", [["lag_1", "lead_2"], ("lag_1", "lead_2")])
    def test_multi_element_offset_type_rejected(self, offset_type: Any) -> None:
        result = OffsetFeatureGroup.match_feature_group_criteria("my_result", self._options(offset_type), None)
        assert result is False, f"Config path should reject: {offset_type!r}"


class TestDigitLikeOffsetSuffixes:
    """A suffix that str.isdigit accepts is not automatically an int; both paths must reject it without raising."""

    def _pattern_options(self) -> Options:
        return Options(context={"partition_by": ["region"], "order_by": "value_int"})

    def _config_options(self, offset_type: str) -> Options:
        return Options(
            context={
                "offset_type": offset_type,
                "in_features": "value_int",
                "partition_by": ["region"],
                "order_by": "value_int",
            }
        )

    @pytest.mark.parametrize("feature_name", ["value_int__lag_²_offset", "value_int__pct_change_²_offset"])
    def test_name_path_rejects_superscript_digit(self, feature_name: str) -> None:
        """Superscript two is isdigit-true but int()-invalid."""
        result = OffsetFeatureGroup.match_feature_group_criteria(feature_name, self._pattern_options(), None)
        assert result is False, f"Name path should reject: {feature_name}"

    @pytest.mark.parametrize("offset_type", ["lag_²", "pct_change_²"])
    def test_config_path_rejects_superscript_digit(self, offset_type: str) -> None:
        result = OffsetFeatureGroup.match_feature_group_criteria("my_result", self._config_options(offset_type), None)
        assert result is False, f"Config path should reject: {offset_type}"

    def test_name_path_rejects_non_ascii_digit(self) -> None:
        """Arabic-Indic three is isdigit-true and int()-valid, but still not an ASCII offset suffix."""
        options = self._pattern_options()
        result = OffsetFeatureGroup.match_feature_group_criteria("value_int__lag_٣_offset", options, None)
        assert result is False

    def test_config_path_rejects_non_ascii_digit(self) -> None:
        result = OffsetFeatureGroup.match_feature_group_criteria("my_result", self._config_options("lag_٣"), None)
        assert result is False

    @pytest.mark.parametrize("offset_type", ["lag_²", "pct_change_²", "lag_٣"])
    def test_supports_offset_type_rejects_digit_like_suffixes(self, offset_type: str) -> None:
        assert not OffsetFeatureGroup._supports_offset_type(offset_type)


class TestHostileInFeatures:
    """A hostile in_features value is a plain non-match; no exception may escape the matcher."""

    @pytest.mark.parametrize("in_features", ["", 0, 3.5, True, {"a": 1}, []])
    def test_rejects_hostile_in_features(self, in_features: Any) -> None:
        options = Options(
            context={
                "offset_type": "lag_1",
                "in_features": in_features,
                "partition_by": ["region"],
                "order_by": "value_int",
            }
        )
        result = OffsetFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False, f"Config path should reject in_features: {in_features!r}"


class TestReturnDataTypeRule:
    """return_data_type_rule should fix the output type only for deterministic ops.

    pct_change_N always returns a fractional ratio (DOUBLE). lag / lead / diff /
    first_value / last_value preserve the input column type, so the rule must
    return None for them.
    """

    def test_pct_change_returns_double(self) -> None:
        feature = Feature(
            "value_int__pct_change_1_offset",
            options=Options(context={"partition_by": ["region"], "order_by": "value_int"}),
        )
        assert OffsetFeatureGroup.return_data_type_rule(feature) == DataType.DOUBLE

    @pytest.mark.parametrize(
        "offset_type",
        ["lag_1", "lead_1", "diff_1", "first_value", "last_value"],
    )
    def test_input_dependent_ops_return_none(self, offset_type: str) -> None:
        feature = Feature(
            f"value_int__{offset_type}_offset",
            options=Options(context={"partition_by": ["region"], "order_by": "value_int"}),
        )
        assert OffsetFeatureGroup.return_data_type_rule(feature) is None


class TestOffsetMatchValidation(MatchValidationTestBase):
    @classmethod
    def feature_group_class(cls) -> Any:
        return OffsetFeatureGroup

    @classmethod
    def valid_operations(cls) -> set[str]:
        return {"first_value", "last_value", "lag_1", "lead_1"}

    @classmethod
    def config_key(cls) -> str:
        return "offset_type"

    @classmethod
    def build_feature_name(cls, operation: str) -> str:
        return f"value_int__{operation}_offset"

    @classmethod
    def build_feature_name_no_source(cls) -> str:
        return "lag_1_offset"

    @classmethod
    def additional_match_options(cls) -> dict[str, Any]:
        return {"in_features": "value_int", "partition_by": ["region"], "order_by": "timestamp"}

    @classmethod
    def pattern_match_options(cls) -> Options:
        return Options(context={"partition_by": ["region"], "order_by": "value_int"})

    @classmethod
    def options_reject_invalid_types(cls) -> bool:
        return True

    @classmethod
    def parity_operations(cls) -> set[str]:
        return {"first_value", "last_value", "lag_1", "lead_2", "diff_1", "pct_change_3"}

    @classmethod
    def malformed_operations(cls) -> set[str]:
        return {"banana", "lag_0", "lag_abc", "lead_"}
