"""Rejection reasons surfaced by ``_strict_validation_rejection_reason``, using each spec's ``expected`` text."""

from __future__ import annotations

from mloda.provider import PropertySpec
from mloda.user import Options

from mloda.community.feature_groups.data_operations.row_preserving.ffill.pyarrow_ffill import PyArrowFfill
from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.pandas_frame_aggregate import (
    PandasFrameAggregate,
)
from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.pyarrow_point_arithmetic import (
    PyArrowPointArithmetic,
)
from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.pyarrow_scalar_arithmetic import (
    PyArrowScalarArithmetic,
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
        reason = PyArrowScalarArithmetic._strict_validation_rejection_reason("value_int__add_constant", options)
        assert reason is not None
        assert "failed validation" in reason
        assert "'constant'" in reason
        assert "'five'" in reason

    def test_nested_singleton_constant_reports_a_reason(self) -> None:
        """``[[5]]`` fails ``constant``'s element_validator, not a guard, so the reason still names it."""
        options = Options(context={"constant": [[5]]})
        reason = PyArrowScalarArithmetic._strict_validation_rejection_reason("value_int__add_constant", options)
        assert reason is not None
        assert "'constant'" in reason

    def test_valid_constant_reports_nothing(self) -> None:
        options = Options(context={"constant": 5})
        assert PyArrowScalarArithmetic._strict_validation_rejection_reason("value_int__add_constant", options) is None

    def test_unrelated_candidate_reports_nothing(self) -> None:
        assert PyArrowScalarArithmetic._strict_validation_rejection_reason("some_unrelated_feature", Options()) is None


class TestConfigPathGuardRejectionReported:
    """A config-path match_guard rejection must be named; a non-candidate must stay silent."""

    def test_mistyped_order_by_reports_a_reason(self) -> None:
        """With partition_by present the guard on order_by is the only failure, so it gets named."""
        options = Options(context={"in_features": "value_float", "order_by": 123, "partition_by": ["region"]})
        reason = PyArrowFfill._strict_validation_rejection_reason("my_result", options)
        assert reason is not None
        assert "option 'order_by' must be" in reason
        assert "got int 123" in reason

    def test_non_candidate_with_guard_rejected_value_reports_nothing(self) -> None:
        """No pattern name and no in_features: ffill was never a candidate, so nothing is reported."""
        options = Options(context={"order_by": 123})
        assert PyArrowFfill._strict_validation_rejection_reason("some_unrelated_feature", options) is None


class TestMultiElementArityRejectionReported:
    """A multi-element container of accepted values is named; the ``expected`` text carries the arity."""

    def test_multi_element_order_by_reports_an_arity_reason(self) -> None:
        """The verdict stays a non-match; only the reason is added."""
        options = Options(
            context={"in_features": "value_float", "order_by": ["ts", "region"], "partition_by": ["region"]}
        )
        assert PyArrowFfill.match_feature_group_criteria("my_result", options, None) is False
        reason = PyArrowFfill._strict_validation_rejection_reason("my_result", options)
        assert reason is not None
        assert "option 'order_by' must be" in reason
        assert "exactly one" in reason
        assert "got list" in reason

    def test_single_element_order_by_reports_nothing(self) -> None:
        """A single-element container is an accepted singleton, so there is nothing to report."""
        options = Options(context={"in_features": "value_float", "order_by": ["ts"], "partition_by": ["region"]})
        assert PyArrowFfill._strict_validation_rejection_reason("my_result", options) is None

    def test_guard_rejected_container_type_reports_the_guard_not_arity(self) -> None:
        """A guard that rejects the container type itself must be named, not misreported as an arity failure."""
        options = Options(context={"arithmetic_op": "add", "in_features": {("a",), ("b",)}})
        reason = PyArrowPointArithmetic._strict_validation_rejection_reason("my_result", options)
        assert reason is not None
        assert "option 'in_features' must be" in reason
        assert "got set" in reason


class TestPresentRequiredOptionReportsNothing:
    """A required option that is present has nothing to report."""

    def test_present_order_by_reports_nothing(self) -> None:
        options = Options(context={"order_by": "ts"})
        assert PyArrowFfill._strict_validation_rejection_reason("value_float__ffill", options) is None


class TestRejectionReasonHookNeverRaises:
    """A value whose repr raises is reported by type name only, never by its text."""

    def test_unreprable_guard_rejected_value_still_reports_a_reason(self) -> None:
        class ExplodingRepr:
            """Value whose repr raises, so reporting must fall back to the type name."""

            def __repr__(self) -> str:
                raise RuntimeError("repr exploded")

        options = Options(context={"order_by": ExplodingRepr()})
        reason = PyArrowFfill._strict_validation_rejection_reason("value_float__ffill", options)
        assert reason is not None
        assert "'order_by'" in reason
        assert "must be" in reason
        assert "got ExplodingRepr" in reason


class TestFrameAggregateNamePathReportsNothing:
    """frame_aggregate parses size and unit from the name, so its name path has nothing to report."""

    def test_rolling_name_skips_option_driven_required_when(self) -> None:
        """The name supplies size and unit, so a frame_type option cannot demand them: the match holds silently."""
        options = Options(context={"partition_by": ["region"], "order_by": "timestamp", "frame_type": "time"})
        assert PandasFrameAggregate.match_feature_group_criteria("sales__sum_rolling_3", options)
        assert PandasFrameAggregate._strict_validation_rejection_reason("sales__sum_rolling_3", options) is None


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
