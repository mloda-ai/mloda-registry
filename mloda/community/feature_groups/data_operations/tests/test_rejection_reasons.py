"""Rejection reasons surfaced by ``_strict_validation_rejection_reason``, using each spec's ``expected`` text."""

from __future__ import annotations

from typing import Any

import pytest
from mloda.provider import PropertySpec, collect_match_rejections
from mloda.user import Options

from mloda.community.feature_groups.data_operations.row_preserving.ema.base import EmaFeatureGroup
from mloda.community.feature_groups.data_operations.row_preserving.ffill.base import FfillFeatureGroup
from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.base import (
    FrameAggregateFeatureGroup,
)
from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.base import (
    PointArithmeticFeatureGroup,
)
from mloda.community.feature_groups.data_operations.row_preserving.rank.base import RankFeatureGroup
from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
    ScalarArithmeticFeatureGroup,
)
from mloda.community.feature_groups.data_operations.row_preserving.sessionization.base import (
    SessionizationFeatureGroup,
)
from mloda.community.feature_groups.data_operations.tests.test_prefix_pattern_collisions import FAMILIES, _load_class


class TestPatternPathGuardRejectionReported:
    """A pattern-path value rejection must be named, not a silent non-match."""

    def test_mistyped_constant_reports_a_reason(self) -> None:
        """``constant`` is validated by element_validator before match_guard ever runs, so this is
        the value-rejection reason core's own hook reports (the same message the real match pass
        records), not a match_guard-worded one.
        """
        options = Options(context={"constant": "five"})
        reason = ScalarArithmeticFeatureGroup._strict_validation_rejection_reason("value_int__add_constant", options)
        assert reason is not None
        assert "failed validation" in reason
        assert "'constant'" in reason
        assert "'five'" in reason

    def test_nested_singleton_constant_reports_a_reason(self) -> None:
        """``[[5]]`` fails ``constant``'s element_validator, not a guard, so the reason still names it."""
        options = Options(context={"constant": [[5]]})
        reason = ScalarArithmeticFeatureGroup._strict_validation_rejection_reason("value_int__add_constant", options)
        assert reason is not None
        assert "'constant'" in reason

    def test_valid_constant_reports_nothing(self) -> None:
        options = Options(context={"constant": 5})
        assert (
            ScalarArithmeticFeatureGroup._strict_validation_rejection_reason("value_int__add_constant", options) is None
        )

    def test_unrelated_candidate_reports_nothing(self) -> None:
        assert (
            ScalarArithmeticFeatureGroup._strict_validation_rejection_reason("some_unrelated_feature", Options())
            is None
        )


class TestConfigPathGuardRejectionReported:
    """A config-path match_guard rejection must be named; a non-candidate must stay silent."""

    def test_mistyped_order_by_reports_a_reason(self) -> None:
        """With the other options present the guard on order_by is the only failure, so it gets named."""
        options = Options(
            context={"rank_type": "row_number", "in_features": "value_float", "order_by": 123, "partition_by": ["g"]}
        )
        reason = RankFeatureGroup._strict_validation_rejection_reason("my_result", options)
        assert reason is not None
        assert "option 'order_by' must be" in reason
        assert "got int 123" in reason

    def test_non_candidate_with_guard_rejected_value_reports_nothing(self) -> None:
        """No pattern name and no in_features: rank was never a candidate, so nothing is reported."""
        options = Options(context={"order_by": 123})
        assert RankFeatureGroup._strict_validation_rejection_reason("some_unrelated_feature", options) is None

    @pytest.mark.parametrize("cls", [EmaFeatureGroup, FfillFeatureGroup, SessionizationFeatureGroup])
    def test_name_only_group_reports_nothing_for_a_config_name(self, cls: Any) -> None:
        """Name-only families: a config-path feature is not a candidate, so no reason is reported."""
        options = Options(context={"in_features": "v", "order_by": 123})
        assert cls._strict_validation_rejection_reason("my_result", options) is None


class TestMultiElementArityRejectionReported:
    """A multi-element container of accepted values is named; the ``expected`` text carries the arity."""

    def test_multi_element_order_by_reports_an_arity_reason(self) -> None:
        """The verdict stays a non-match; only the reason is added."""
        options = Options(
            context={
                "rank_type": "row_number",
                "in_features": "value_float",
                "order_by": ["ts", "region"],
                "partition_by": ["region"],
            }
        )
        assert RankFeatureGroup.match_feature_group_criteria("my_result", options, None) is False
        reason = RankFeatureGroup._strict_validation_rejection_reason("my_result", options)
        assert reason is not None
        assert "option 'order_by' must be" in reason
        assert "exactly one" in reason
        assert "got list" in reason

    def test_single_element_order_by_reports_nothing(self) -> None:
        """A single-element container is an accepted singleton, so there is nothing to report."""
        options = Options(
            context={
                "rank_type": "row_number",
                "in_features": "value_float",
                "order_by": ["ts"],
                "partition_by": ["region"],
            }
        )
        assert RankFeatureGroup._strict_validation_rejection_reason("my_result", options) is None

    def test_guard_rejected_container_type_reports_the_guard_not_arity(self) -> None:
        """A guard that rejects the container type itself must be named, not misreported as an arity failure.

        The set holds exactly two strings so the in_features count gate passes and the guard is reached.
        """
        options = Options(context={"arithmetic_op": "add", "in_features": {"a", "b"}})
        assert PointArithmeticFeatureGroup.match_feature_group_criteria("my_result", options, None) is False
        reason = PointArithmeticFeatureGroup._strict_validation_rejection_reason("my_result", options)
        assert reason is not None
        assert "option 'in_features' must be" in reason
        assert "got set" in reason
        assert "exactly one" not in reason


class TestPresentRequiredOptionReportsNothing:
    """A required option that is present has nothing to report."""

    def test_present_order_by_reports_nothing(self) -> None:
        options = Options(context={"order_by": "ts"})
        assert FfillFeatureGroup._strict_validation_rejection_reason("value_float__ffill", options) is None


class TestRejectionReasonHookNeverRaises:
    """A value whose repr raises is reported by type name only, never by its text."""

    def test_unreprable_guard_rejected_value_still_reports_a_reason(self) -> None:
        class ExplodingRepr:
            """Value whose repr raises, so reporting must fall back to the type name."""

            def __repr__(self) -> str:
                raise RuntimeError("repr exploded")

        options = Options(context={"order_by": ExplodingRepr()})
        reason = FfillFeatureGroup._strict_validation_rejection_reason("value_float__ffill", options)
        assert reason is not None
        assert "'order_by'" in reason
        assert "must be" in reason
        assert "got ExplodingRepr" in reason


class TestFrameAggregateNamePathReportsNothing:
    """frame_aggregate parses size and unit from the name, so its name path has nothing to report."""

    def test_rolling_name_skips_option_driven_required_when(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The name supplies size and unit, so a frame_type option cannot demand them: the match holds silently."""
        # The env downgrade is set because the stray frame_type now contradicts the name.
        monkeypatch.setenv("MLODA_ALLOW_FORWARDED_NAME_MISMATCH", "1")
        options = Options(context={"partition_by": ["region"], "order_by": "timestamp", "frame_type": "time"})
        assert FrameAggregateFeatureGroup.match_feature_group_criteria("sales__sum_rolling_3", options)
        assert FrameAggregateFeatureGroup._strict_validation_rejection_reason("sales__sum_rolling_3", options) is None


class TestFrameAggregateInheritedFrameTypeReported:
    """An inherited frame_type declines frame_aggregate's config path with a reason naming the fix."""

    def test_inherited_group_frame_type_records_a_reason(self) -> None:
        options = Options(context={"aggregation_type": "sum", "in_features": "v", "partition_by": ["g"]})
        options.inherit_from(Options(group={"frame_type": "rolling"}))
        with collect_match_rejections() as reasons:
            assert FrameAggregateFeatureGroup.match_feature_group_criteria("my_result", options, None) is False
        reason = reasons[FrameAggregateFeatureGroup.get_class_name()].reason
        assert "frame_type" in reason
        assert "inherited" in reason


class TestEveryGuardedSpecDeclaresExpected:
    """A guarded PROPERTY_MAPPING spec without ``expected`` silently falls back to a bare non-match."""

    def test_every_guarded_spec_declares_expected(self) -> None:
        offenders: list[str] = []
        for family in FAMILIES:
            cls = _load_class(family)
            property_mapping = getattr(cls, "PROPERTY_MAPPING", {})
            for key, spec in property_mapping.items():
                if not isinstance(spec, PropertySpec):
                    continue
                if spec.match_guard is not None and spec.expected is None:
                    offenders.append(f"{family.key}.{key}")
        assert offenders == [], f"Guarded specs missing 'expected': {offenders}"


class TestReleasedLeafImportCompat:
    """Released leaves still import the old name."""

    def test_rejection_reason_mixin_is_feature_chain_parser_mixin(self) -> None:
        from mloda.provider import FeatureChainParserMixin

        from mloda.community.feature_groups.data_operations.base import RejectionReasonMixin

        assert RejectionReasonMixin is FeatureChainParserMixin

    def test_deprecated_input_columns_and_framework_shim_still_works(self) -> None:
        """Old leaves pinned against >= this base still call this removed hook."""
        from mloda.community.feature_groups.data_operations.python_dict_helpers import (
            input_columns_and_framework,
        )

        assert input_columns_and_framework({"a": [1]}) == (["a"], "PythonDict")

    def test_aggregation_base_is_still_a_feature_group(self) -> None:
        """Released aggregation-family leaves subclass ``AggregationFeatureGroupBase`` alone and must stay FeatureGroups."""
        from mloda.provider import FeatureGroup

        from mloda.community.feature_groups.data_operations.aggregation_base import AggregationFeatureGroupBase

        assert issubclass(AggregationFeatureGroupBase, FeatureGroup)
