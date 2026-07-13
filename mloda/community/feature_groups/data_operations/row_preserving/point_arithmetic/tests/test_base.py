"""Tests for PointArithmeticFeatureGroup base class."""

from __future__ import annotations

from typing import Any

import pytest

from mloda.core.abstract_plugins.components.feature_name import FeatureName
from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.match_validation import MatchValidationTestBase
from mloda.user import Feature

from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.base import (
    ARITHMETIC_OPERATIONS,
    PointArithmeticFeatureGroup,
)


class TestClassAttributes:
    def test_prefix_pattern_exists(self) -> None:
        assert hasattr(PointArithmeticFeatureGroup, "PREFIX_PATTERN")
        assert isinstance(PointArithmeticFeatureGroup.PREFIX_PATTERN, str)

    def test_arithmetic_operations_contains_all_operations(self) -> None:
        expected_ops = {"add", "subtract", "multiply", "divide"}
        for op in expected_ops:
            assert op in ARITHMETIC_OPERATIONS, f"Missing operation: {op}"

    def test_arithmetic_operations_has_no_extra_operations(self) -> None:
        """Only the four canonical operations are defined."""
        assert set(ARITHMETIC_OPERATIONS.keys()) == {"add", "subtract", "multiply", "divide"}

    def test_min_in_features_is_two(self) -> None:
        assert PointArithmeticFeatureGroup.MIN_IN_FEATURES == 2

    def test_max_in_features_is_two(self) -> None:
        assert PointArithmeticFeatureGroup.MAX_IN_FEATURES == 2

    def test_arithmetic_op_constant(self) -> None:
        assert PointArithmeticFeatureGroup.ARITHMETIC_OP == "arithmetic_op"


class TestPatternMatching:
    @pytest.mark.parametrize(
        "feature_name",
        [
            "value_int&amount__add_point",
            "value_int&amount__subtract_point",
            "value_int&amount__multiply_point",
            "value_int&amount__divide_point",
        ],
    )
    def test_matches_all_operations(self, feature_name: str) -> None:
        options = Options()
        result = PointArithmeticFeatureGroup.match_feature_group_criteria(feature_name, options, None)
        assert result is True, f"Should match: {feature_name}"

    def test_no_match_wrong_suffix(self) -> None:
        options = Options()
        result = PointArithmeticFeatureGroup.match_feature_group_criteria(
            "value_int&amount__add_constant", options, None
        )
        assert result is False

    def test_no_match_no_suffix(self) -> None:
        options = Options()
        result = PointArithmeticFeatureGroup.match_feature_group_criteria("value_int&amount__add", options, None)
        assert result is False

    def test_no_match_no_source_column(self) -> None:
        options = Options()
        result = PointArithmeticFeatureGroup.match_feature_group_criteria("add_point", options, None)
        assert result is False

    def test_no_match_invalid_operation(self) -> None:
        options = Options()
        result = PointArithmeticFeatureGroup.match_feature_group_criteria(
            "value_int&amount__unknown_point", options, None
        )
        assert result is False


class TestPatternParsing:
    def test_parse_add_operation(self) -> None:
        operation = PointArithmeticFeatureGroup.get_arithmetic_op("value_int&amount__add_point")
        assert operation == "add"

    def test_parse_subtract_operation(self) -> None:
        operation = PointArithmeticFeatureGroup.get_arithmetic_op("value_int&amount__subtract_point")
        assert operation == "subtract"

    def test_parse_multiply_operation(self) -> None:
        operation = PointArithmeticFeatureGroup.get_arithmetic_op("value_int&amount__multiply_point")
        assert operation == "multiply"

    def test_parse_divide_operation(self) -> None:
        operation = PointArithmeticFeatureGroup.get_arithmetic_op("value_int&amount__divide_point")
        assert operation == "divide"

    def test_parse_source_features(self) -> None:
        feature = Feature("value_int&amount__add_point", options=Options())
        source_features = PointArithmeticFeatureGroup._extract_source_features(feature)
        assert source_features == ["value_int", "amount"]

    def test_parse_source_features_with_underscores(self) -> None:
        feature = Feature("my_a&my_b__multiply_point", options=Options())
        source_features = PointArithmeticFeatureGroup._extract_source_features(feature)
        assert source_features == ["my_a", "my_b"]

    def test_greedy_regex_for_chained_op_tokens(self) -> None:
        """Pin the greedy-parse contract shared with sibling families.

        ``rsplit("__", 1)`` plus the greedy ``.*__([\\w]+)_point$`` pattern
        means that for a chained name like ``value_int&amount__add__subtract_point``
        the source string before the last ``__`` is ``value_int&amount__add`` and the
        captured op token is ``subtract``. Splitting on the ``&`` IN_FEATURE_SEPARATOR
        then yields ``["value_int", "amount__add"]``.

        A future regex tightening must be a deliberate decision; this test
        exists to surface any silent change to the parse contract.
        """
        feature = Feature("value_int&amount__add__subtract_point", options=Options())
        assert PointArithmeticFeatureGroup._extract_source_features(feature) == ["value_int", "amount__add"]
        assert PointArithmeticFeatureGroup._extract_arithmetic_op(feature) == "subtract"


class TestConfigBasedFeatures:
    def test_config_based_match(self) -> None:
        options = Options(
            context={
                "arithmetic_op": "add",
                "in_features": ["value_int", "amount"],
            }
        )
        result = PointArithmeticFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is True

    def test_config_based_match_rejects_invalid_op(self) -> None:
        options = Options(
            context={
                "arithmetic_op": "invalid_op",
                "in_features": ["value_int", "amount"],
            }
        )
        result = PointArithmeticFeatureGroup.match_feature_group_criteria("my_result", options, None)
        assert result is False


class TestTwoColumnEnforcement:
    """Verify that MIN_IN_FEATURES=2 / MAX_IN_FEATURES=2 enforces two-column behavior."""

    def test_min_in_features_is_two(self) -> None:
        assert PointArithmeticFeatureGroup.MIN_IN_FEATURES == 2

    def test_max_in_features_is_two(self) -> None:
        assert PointArithmeticFeatureGroup.MAX_IN_FEATURES == 2

    def test_input_features_rejects_one_option_in_feature(self) -> None:
        options = Options(
            context={
                "arithmetic_op": "add",
                "in_features": ["value_int"],
            }
        )
        instance = PointArithmeticFeatureGroup()
        with pytest.raises(ValueError, match="at least 2"):
            instance.input_features(options, FeatureName("my_result"))

    def test_input_features_rejects_three_option_in_features(self) -> None:
        options = Options(
            context={
                "arithmetic_op": "add",
                "in_features": ["a", "b", "c"],
            }
        )
        instance = PointArithmeticFeatureGroup()
        with pytest.raises(ValueError, match="at most 2"):
            instance.input_features(options, FeatureName("my_result"))

    def test_input_features_returns_two_features_for_string_pattern(self) -> None:
        instance = PointArithmeticFeatureGroup()
        result = instance.input_features(Options(), FeatureName("value_int&amount__add_point"))
        assert result is not None
        assert len(result) == 2
        names = {f.name for f in result}
        assert names == {"value_int", "amount"}

    def test_input_features_returns_two_features_for_option_config(self) -> None:
        options = Options(
            context={
                "arithmetic_op": "add",
                "in_features": ["revenue", "cost"],
            }
        )
        instance = PointArithmeticFeatureGroup()
        result = instance.input_features(options, FeatureName("my_diff"))
        assert result is not None
        assert len(result) == 2
        names = {f.name for f in result}
        assert names == {"revenue", "cost"}


class TestArithmeticOpExtraction:
    """Verify arithmetic op extraction from both string and option sources."""

    def test_get_arithmetic_op_raises_for_non_pattern_name(self) -> None:
        with pytest.raises(ValueError, match="Could not extract"):
            PointArithmeticFeatureGroup.get_arithmetic_op("plain_name")

    def test_extract_arithmetic_op_from_options(self) -> None:
        options = Options(
            context={
                "arithmetic_op": "divide",
                "in_features": ["value_int", "amount"],
            }
        )
        feature = Feature("my_result", options=options)
        op = PointArithmeticFeatureGroup._extract_arithmetic_op(feature)
        assert op == "divide"

    def test_extract_arithmetic_op_raises_without_option(self) -> None:
        feature = Feature("plain_name", options=Options())
        with pytest.raises(ValueError, match="Could not extract"):
            PointArithmeticFeatureGroup._extract_arithmetic_op(feature)

    @pytest.mark.parametrize("op", list(ARITHMETIC_OPERATIONS.keys()))
    def test_get_arithmetic_op_for_all_ops(self, op: str) -> None:
        feature_name = f"col_a&col_b__{op}_point"
        result = PointArithmeticFeatureGroup.get_arithmetic_op(feature_name)
        assert result == op


class TestUnorderedInFeaturesRejected:
    """Unordered ``in_features`` (set/frozenset) must be rejected.

    Operand order (col_a, col_b) is undefined for an unordered collection,
    which is wrong for non-commutative ops (subtract/divide). The raw
    ``in_features`` option must therefore be an ordered list/tuple; a set or
    frozenset must raise ValueError in ``_extract_source_features``.
    """

    @pytest.mark.parametrize(
        "in_features",
        [
            {"value_int", "amount"},
            frozenset({"value_int", "amount"}),
        ],
    )
    def test_extract_source_features_rejects_unordered_in_features(self, in_features: Any) -> None:
        feature = Feature(
            "my_result",
            options=Options(
                context={
                    "arithmetic_op": "subtract",
                    "in_features": in_features,
                }
            ),
        )
        with pytest.raises(ValueError, match="ordered"):
            PointArithmeticFeatureGroup._extract_source_features(feature)


class TestMalformedInFeaturesShapeRejectedAtMatch:
    """Unordered ``in_features`` must fail to match, not match-then-fail-late.

    ``in_features`` carries a ``match_guard`` that only accepts ordered
    containers (list/tuple). A hashable but unordered collection (frozenset)
    must therefore cause ``match_feature_group_criteria`` to return False up
    front, since operand order is significant for subtract/divide; before the
    validator was added it matched as True. A valid ordered list must still
    match (control).

    A mapping (dict) is intentionally not asserted here: an unhashable value
    raises ``TypeError`` inside the core property-mapping parser before the
    ``match_guard`` runs, independently of this feature group. The mapping
    case is pinned at the extract layer instead (see
    ``TestNonOrderedInFeaturesRejectedOnExtract``).
    """

    def test_match_rejects_unordered_frozenset_in_features(self) -> None:
        options = Options(
            context={
                "arithmetic_op": "subtract",
                "in_features": frozenset({"value_int", "amount"}),
            }
        )
        result = PointArithmeticFeatureGroup.match_feature_group_criteria("my_diff", options, None)
        assert result is False

    def test_match_accepts_ordered_list_in_features(self) -> None:
        options = Options(
            context={
                "arithmetic_op": "subtract",
                "in_features": ["value_int", "amount"],
            }
        )
        result = PointArithmeticFeatureGroup.match_feature_group_criteria("my_diff", options, None)
        assert result is True


class TestNonOrderedInFeaturesRejectedOnExtract:
    """The ``_extract_source_features`` fallback must match its own error message.

    Its message promises an "ordered list or tuple"; arbitrary iterables and
    mappings (dict, generator) must therefore raise ValueError rather than
    being silently iterated, defending the contract at compute time.
    """

    def test_extract_source_features_rejects_mapping(self) -> None:
        feature = Feature(
            "my_result",
            options=Options(
                context={
                    "arithmetic_op": "subtract",
                    "in_features": {"value_int": 1, "amount": 2},
                }
            ),
        )
        with pytest.raises(ValueError, match="ordered list or tuple"):
            PointArithmeticFeatureGroup._extract_source_features(feature)

    def test_extract_source_features_rejects_generator(self) -> None:
        def gen() -> Any:
            yield "value_int"
            yield "amount"

        feature = Feature(
            "my_result",
            options=Options(
                context={
                    "arithmetic_op": "subtract",
                    "in_features": gen(),
                }
            ),
        )
        with pytest.raises(ValueError, match="ordered list or tuple"):
            PointArithmeticFeatureGroup._extract_source_features(feature)


class TestPointArithmeticMatchValidation(MatchValidationTestBase):
    """Shared match-validation tests adapted for point arithmetic."""

    @classmethod
    def feature_group_class(cls) -> Any:
        return PointArithmeticFeatureGroup

    @classmethod
    def valid_operations(cls) -> set[str]:
        return set(ARITHMETIC_OPERATIONS)

    @classmethod
    def config_key(cls) -> str:
        return "arithmetic_op"

    @classmethod
    def build_feature_name(cls, operation: str) -> str:
        return f"value_int&amount__{operation}_point"

    @classmethod
    def build_feature_name_no_source(cls) -> str:
        return "add_point"

    @classmethod
    def additional_match_options(cls) -> dict[str, Any]:
        return {"in_features": ["value_int", "amount"]}
