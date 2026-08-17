"""Tests for RankFeatureGroup base class."""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.options import Options
from mloda.provider import DefaultOptionKeys
from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
from mloda.testing.feature_groups.data_operations.helpers import extract_column, feature_set_for
from mloda.testing.feature_groups.data_operations.match_validation import MatchValidationTestBase, TokenCase
from mloda.testing.feature_groups.data_operations.row_preserving.rank.reference import ReferenceRank
from mloda.user import DataType, Feature

from mloda.community.feature_groups.data_operations.row_preserving.rank.base import (
    RankFeatureGroup,
)


class TestClassAttributes:
    """Tests for RankFeatureGroup class attributes."""

    def test_prefix_pattern_exists(self) -> None:
        assert hasattr(RankFeatureGroup, "PREFIX_PATTERN")
        assert isinstance(RankFeatureGroup.PREFIX_PATTERN, str)

    def test_rank_types_exists(self) -> None:
        assert hasattr(RankFeatureGroup, "RANK_TYPES")
        assert isinstance(RankFeatureGroup.RANK_TYPES, dict)

    def test_rank_types_contains_standard_operations(self) -> None:
        expected_ops = {"row_number", "rank", "dense_rank", "percent_rank"}
        for op in expected_ops:
            assert op in RankFeatureGroup.RANK_TYPES, f"Missing rank type: {op}"

    @pytest.mark.parametrize(
        "rank_types",
        [
            pytest.param(["ntile_4", "ntile_10"], id="ntile"),
            pytest.param(["top_5", "top_1"], id="top_n"),
            pytest.param(["bottom_5", "bottom_1"], id="bottom_n"),
        ],
    )
    def test_supports_parametric_rank_types(self, rank_types: list[str]) -> None:
        for rank_type in rank_types:
            assert RankFeatureGroup._supports_rank_type(rank_type)

    @pytest.mark.parametrize(
        "rank_types",
        [
            pytest.param(["ntile_0", "ntile_abc"], id="ntile"),
            pytest.param(["top_0", "top_abc"], id="top_n"),
            pytest.param(["bottom_0", "bottom_abc"], id="bottom_n"),
        ],
    )
    def test_rejects_invalid_parametric_rank_types(self, rank_types: list[str]) -> None:
        for rank_type in rank_types:
            assert not RankFeatureGroup._supports_rank_type(rank_type)

    def test_supports_ntile_1(self) -> None:
        assert RankFeatureGroup._supports_rank_type("ntile_1")

    def test_rejects_ntile_negative(self) -> None:
        assert not RankFeatureGroup._supports_rank_type("ntile_-1")

    def test_min_in_features_is_one(self) -> None:
        assert RankFeatureGroup.MIN_IN_FEATURES == 1

    def test_max_in_features_is_one(self) -> None:
        assert RankFeatureGroup.MAX_IN_FEATURES == 1


class TestPropertyMapping:
    """Tests for PROPERTY_MAPPING consistency with context parameter discovery."""

    def test_property_mapping_contains_partition_by(self) -> None:
        mapping = RankFeatureGroup.PROPERTY_MAPPING
        assert RankFeatureGroup.PARTITION_BY in mapping
        entry = mapping[RankFeatureGroup.PARTITION_BY]
        assert entry.context is True

    def test_property_mapping_contains_order_by(self) -> None:
        mapping = RankFeatureGroup.PROPERTY_MAPPING
        assert RankFeatureGroup.ORDER_BY in mapping
        entry = mapping[RankFeatureGroup.ORDER_BY]
        assert entry.context is True

    def test_property_mapping_contains_rank_type(self) -> None:
        mapping = RankFeatureGroup.PROPERTY_MAPPING
        assert RankFeatureGroup.RANK_TYPE in mapping
        entry = mapping[RankFeatureGroup.RANK_TYPE]
        assert entry.context is True

    def test_property_mapping_contains_in_features(self) -> None:
        mapping = RankFeatureGroup.PROPERTY_MAPPING
        assert DefaultOptionKeys.in_features in mapping
        entry = mapping[DefaultOptionKeys.in_features]
        assert entry.context is True


class TestPatternMatching:
    """Tests for feature name pattern matching."""

    @pytest.mark.parametrize(
        "feature_name",
        [
            "value_int__row_number_ranked",
            "value_int__rank_ranked",
            "value_int__dense_rank_ranked",
            "value_int__percent_rank_ranked",
        ],
    )
    def test_matches_standard_rank_types(self, feature_name: str) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "value_int"})
        result = RankFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is True, f"Should match: {feature_name}"

    @pytest.mark.parametrize(
        ("feature_name", "expected"),
        [
            pytest.param("value_int__ntile_4_ranked", True, id="ntile"),
            pytest.param("value_int__rank_window", False, id="wrong_suffix"),
            pytest.param("rank_ranked", False, id="no_source_column"),
            pytest.param("value_int__unknown_ranked", False, id="invalid_rank_type"),
            pytest.param("value_int__top_5_ranked", True, id="top_n"),
            pytest.param("value_int__bottom_3_ranked", True, id="bottom_n"),
        ],
    )
    def test_match_by_name(self, feature_name: str, expected: bool) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "value_int"})
        result = RankFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is expected


class TestPatternParsing:
    """Tests for extracting rank type and source column."""

    @pytest.mark.parametrize(
        ("feature_name", "expected"),
        [
            pytest.param("value_int__row_number_ranked", "row_number", id="row_number"),
            pytest.param("my_col__dense_rank_ranked", "dense_rank", id="dense_rank"),
            pytest.param("value_int__ntile_4_ranked", "ntile_4", id="ntile"),
            pytest.param("value_int__top_5_ranked", "top_5", id="top_n"),
            pytest.param("value_int__bottom_3_ranked", "bottom_3", id="bottom_n"),
        ],
    )
    def test_parse_rank_type(self, feature_name: str, expected: str) -> None:
        rank_type = RankFeatureGroup.get_rank_type(feature_name)
        assert rank_type == expected

    def test_parse_source_feature(self) -> None:
        from mloda.user import Feature

        feature = Feature(
            "value_int__rank_ranked",
            options=Options(context={"partition_by": ["region"], "order_by": "value_int"}),
        )
        source_features = RankFeatureGroup._extract_source_features(feature)
        assert source_features == ["value_int"]


class TestConfigValidation:
    """Tests for partition_by and order_by validation."""

    def test_partition_by_required(self) -> None:
        options = Options(context={"order_by": "value_int"})
        result = RankFeatureGroup.match_feature_group_criteria("value_int__rank_ranked", options, None)
        assert result is False

    def test_order_by_required(self) -> None:
        options = Options(context={"partition_by": ["region"]})
        result = RankFeatureGroup.match_feature_group_criteria("value_int__rank_ranked", options, None)
        assert result is False

    def test_partition_by_must_be_list_or_tuple(self) -> None:
        options = Options(context={"partition_by": "region", "order_by": "value_int"})
        result = RankFeatureGroup.match_feature_group_criteria("value_int__rank_ranked", options, None)
        assert result is False

    def test_partition_by_accepts_tuple(self) -> None:
        options = Options(context={"partition_by": ("region",), "order_by": "value_int"})
        result = RankFeatureGroup.match_feature_group_criteria("value_int__rank_ranked", options, None)
        assert result is True

    def test_partition_by_rejects_empty_list(self) -> None:
        options = Options(context={"partition_by": [], "order_by": "value_int"})
        result = RankFeatureGroup.match_feature_group_criteria("value_int__rank_ranked", options, None)
        assert result is False

    def test_partition_by_rejects_empty_tuple(self) -> None:
        options = Options(context={"partition_by": (), "order_by": "value_int"})
        result = RankFeatureGroup.match_feature_group_criteria("value_int__rank_ranked", options, None)
        assert result is False

    def test_order_by_rejects_multiple_columns(self) -> None:
        # order_by names ONE column; a one-element container is valid caller syntax for it
        # (see TestOrderByArity), so only a genuinely multi-valued container is invalid here.
        options = Options(context={"partition_by": ["region"], "order_by": ["value_int", "region"]})
        result = RankFeatureGroup.match_feature_group_criteria("value_int__rank_ranked", options, None)
        assert result is False

    def test_order_by_rejects_non_string(self) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": 42})
        result = RankFeatureGroup.match_feature_group_criteria("value_int__rank_ranked", options, None)
        assert result is False

    def test_valid_config(self) -> None:
        options = Options(context={"partition_by": ["region", "category"], "order_by": "value_int"})
        result = RankFeatureGroup.match_feature_group_criteria("value_int__rank_ranked", options, None)
        assert result is True


class TestConfigBasedFeatures:
    """Tests for configuration-based feature matching."""

    def test_config_based_match(self) -> None:
        options = Options(
            context={
                "rank_type": "row_number",
                "in_features": "value_int",
                "partition_by": ["region"],
                "order_by": "value_int",
            }
        )
        result = RankFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is True

    def test_config_based_match_rejects_missing_order_by(self) -> None:
        options = Options(
            context={
                "rank_type": "row_number",
                "in_features": "value_int",
                "partition_by": ["region"],
            }
        )
        result = RankFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False

    def test_config_based_calculate_feature(self) -> None:
        import pyarrow as pa

        from mloda.core.abstract_plugins.components.feature_set import FeatureSet
        from mloda.testing.feature_groups.data_operations.row_preserving.rank.reference import ReferenceRank
        from mloda.testing.data_creator.pyarrow import PyArrowDataOpsTestDataCreator
        from mloda.user import Feature

        table = PyArrowDataOpsTestDataCreator.create()

        feature = Feature(
            "my_rank_result",
            options=Options(
                context={
                    "rank_type": "row_number",
                    "in_features": "value_int",
                    "partition_by": ["region"],
                    "order_by": "value_int",
                }
            ),
        )
        fs = FeatureSet()
        fs.add(feature)

        result = ReferenceRank.calculate_feature(table, fs)
        assert isinstance(result, pa.Table)
        assert "my_rank_result" in result.column_names
        assert result.num_rows == 12


class TestConfigBasedParametricRankTypes:
    """The config path must accept exactly the rank types the feature-name path accepts.

    ``ntile_N`` / ``top_N`` / ``bottom_N`` are supported through the name path, so the
    ``rank_type`` declaration must admit them on the config path as well.
    """

    def _options(self, rank_type: Any) -> Options:
        return Options(
            context={
                "rank_type": rank_type,
                "in_features": "value_int",
                "partition_by": ["region"],
                "order_by": "value_int",
            }
        )

    @pytest.mark.parametrize("rank_type", ["ntile_4", "top_5", "bottom_3"])
    def test_config_based_match_accepts_parametric_rank_types(self, rank_type: str) -> None:
        result = RankFeatureGroup.match_feature_group_criteria("my_result", self._options(rank_type), None)
        assert result is True, f"Config path should accept: {rank_type}"

    @pytest.mark.parametrize(
        "rank_type",
        ["ntile_0", "top_0", "bottom_0", "ntile_-1", "ntile_abc", "lag_1", "banana"],
    )
    def test_config_based_match_rejects_invalid_rank_types(self, rank_type: str) -> None:
        result = RankFeatureGroup.match_feature_group_criteria("my_result", self._options(rank_type), None)
        assert result is False, f"Config path should reject: {rank_type}"

    @pytest.mark.parametrize("rank_type", [42, {"rank_type": "row_number"}])
    def test_config_based_match_rejects_non_string_rank_type(self, rank_type: Any) -> None:
        """A non-string rank_type is a plain non-match, never an uncaught exception."""
        result = RankFeatureGroup.match_feature_group_criteria("my_result", self._options(rank_type), None)
        assert result is False, f"Config path should reject: {rank_type!r}"


class TestDigitLikeRankSuffixes:
    """A suffix that str.isdigit accepts is not automatically an int; both paths must reject it without raising."""

    def _pattern_options(self) -> Options:
        return Options(context={"partition_by": ["region"], "order_by": "value_int"})

    def _config_options(self, rank_type: str) -> Options:
        return Options(
            context={
                "rank_type": rank_type,
                "in_features": "value_int",
                "partition_by": ["region"],
                "order_by": "value_int",
            }
        )

    @pytest.mark.parametrize("feature_name", ["value_int__ntile_²_ranked", "value_int__top_²_ranked"])
    def test_name_path_rejects_superscript_digit(self, feature_name: str) -> None:
        """Superscript two is isdigit-true but int()-invalid."""
        result = RankFeatureGroup.match_feature_group_criteria(feature_name, self._pattern_options(), None)
        assert result is False, f"Name path should reject: {feature_name}"

    @pytest.mark.parametrize("rank_type", ["ntile_²", "top_²"])
    def test_config_path_rejects_superscript_digit(self, rank_type: str) -> None:
        result = RankFeatureGroup.match_feature_group_criteria("my_result", self._config_options(rank_type), None)
        assert result is False, f"Config path should reject: {rank_type}"

    def test_name_path_rejects_non_ascii_digit(self) -> None:
        """Arabic-Indic three is isdigit-true and int()-valid, but still not an ASCII rank suffix."""
        options = self._pattern_options()
        result = RankFeatureGroup.match_feature_group_criteria("value_int__ntile_٣_ranked", options, None)
        assert result is False

    def test_config_path_rejects_non_ascii_digit(self) -> None:
        result = RankFeatureGroup.match_feature_group_criteria("my_result", self._config_options("ntile_٣"), None)
        assert result is False

    @pytest.mark.parametrize("rank_type", ["ntile_²", "top_²", "ntile_٣"])
    def test_supports_rank_type_rejects_digit_like_suffixes(self, rank_type: str) -> None:
        assert not RankFeatureGroup._supports_rank_type(rank_type)


class TestHostileInFeatures:
    """A hostile in_features value is a plain non-match; no exception may escape the matcher."""

    @pytest.mark.parametrize("in_features", ["", 0, 3.5, True, {"a": 1}, []])
    def test_rejects_hostile_in_features(self, in_features: Any) -> None:
        options = Options(
            context={
                "rank_type": "row_number",
                "in_features": in_features,
                "partition_by": ["region"],
                "order_by": "value_int",
            }
        )
        result = RankFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False, f"Config path should reject in_features: {in_features!r}"


class TestReturnDataTypeRule:
    """return_data_type_rule should fix the output type for deterministic rank ops.

    row_number / rank / dense_rank / ntile_N are integer ranks (INT64).
    percent_rank is a fractional rank (DOUBLE). top_N / bottom_N are deferred
    and not yet declared, so the rule must return None for them.
    """

    @pytest.mark.parametrize("rank_type", ["row_number", "rank", "dense_rank", "ntile_4"])
    def test_integer_rank_ops_return_int64(self, rank_type: str) -> None:
        feature = Feature(
            f"value_int__{rank_type}_ranked",
            options=Options(context={"partition_by": ["region"], "order_by": "value_int"}),
        )
        assert RankFeatureGroup.return_data_type_rule(feature) == DataType.INT64

    def test_percent_rank_returns_double(self) -> None:
        feature = Feature(
            "value_int__percent_rank_ranked",
            options=Options(context={"partition_by": ["region"], "order_by": "value_int"}),
        )
        assert RankFeatureGroup.return_data_type_rule(feature) == DataType.DOUBLE

    @pytest.mark.parametrize("rank_type", ["top_3", "bottom_3"])
    def test_deferred_ops_return_none(self, rank_type: str) -> None:
        feature = Feature(
            f"value_int__{rank_type}_ranked",
            options=Options(context={"partition_by": ["region"], "order_by": "value_int"}),
        )
        assert RankFeatureGroup.return_data_type_rule(feature) is None


class TestForwardedRankTypeMismatch:
    """A group-forwarded ``rank_type`` that contradicts the name-parsed type must be rejected, not silently ignored."""

    def test_mismatched_forwarded_rank_type_raises(self) -> None:
        consumer_options = Options(group={"rank_type": "ntile_8"})
        child_options = Options(context={"partition_by": ["region"], "order_by": "value_int"})
        child_options.inherit_from(consumer_options)

        with pytest.raises(ValueError, match="rank_type"):
            RankFeatureGroup.match_feature_group_criteria("value_int__ntile_4_ranked", child_options, None)

    def test_matching_forwarded_rank_type_is_accepted(self) -> None:
        consumer_options = Options(group={"rank_type": "ntile_4"})
        child_options = Options(context={"partition_by": ["region"], "order_by": "value_int"})
        child_options.inherit_from(consumer_options)

        result = RankFeatureGroup.match_feature_group_criteria("value_int__ntile_4_ranked", child_options, None)
        assert result is True

    def test_mismatched_forwarded_fixed_rank_type_raises(self) -> None:
        consumer_options = Options(group={"rank_type": "rank"})
        child_options = Options(context={"partition_by": ["region"], "order_by": "value_int"})
        child_options.inherit_from(consumer_options)

        with pytest.raises(ValueError, match="rank_type"):
            RankFeatureGroup.match_feature_group_criteria("value_int__dense_rank_ranked", child_options, None)


class TestRankMatchValidation(MatchValidationTestBase):
    @classmethod
    def feature_group_class(cls) -> Any:
        return RankFeatureGroup

    @classmethod
    def valid_operations(cls) -> set[str]:
        return set(RankFeatureGroup.RANK_TYPES)

    @classmethod
    def config_key(cls) -> str:
        return "rank_type"

    @classmethod
    def build_feature_name(cls, operation: str) -> str:
        return f"value_int__{operation}_ranked"

    @classmethod
    def build_feature_name_no_source(cls) -> str:
        return "row_number_ranked"

    @classmethod
    def additional_match_options(cls) -> dict[str, Any]:
        return {"in_features": "value_int", "partition_by": ["region"], "order_by": "value_int"}

    @classmethod
    def pattern_match_options(cls) -> Options:
        return Options(context={"partition_by": ["region"], "order_by": "value_int"})

    @classmethod
    def parity_operations(cls) -> set[str]:
        # Widen, never narrow: valid_operations() covers the fixed rank types only, so add one
        # instance of each parametric family (ntile_N / top_N / bottom_N) on top of it.
        return cls.valid_operations() | {"ntile_4", "top_5", "bottom_3"}

    @classmethod
    def malformed_operations(cls) -> set[str]:
        return {"ntile_0", "ntile_abc", "top_0", "bottom_0", "banana"}

    @classmethod
    def token_cases(cls) -> list[TokenCase]:
        # order_by names one column, and rank requires it on both paths.
        return [*super().token_cases(), TokenCase("order_by", "value_int", "region")]

    @classmethod
    def dispatch_values(cls, options: Options) -> list[Any]:
        return [RankFeatureGroup._extract_rank_type(Feature("my_result", options=options))]

    @classmethod
    def compute_values(cls, options: Options) -> list[Any] | None:
        # Rank reads order_by inline in calculate_feature rather than through an extractor,
        # so only a run through the backend shows a container reaching it unwrapped.
        result = ReferenceRank.calculate_feature(
            PyArrowDataOpsTestDataCreator.create(), feature_set_for("my_rank_result", options)
        )
        return extract_column(result, "my_rank_result")
