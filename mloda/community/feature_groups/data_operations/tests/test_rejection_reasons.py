"""Rejection reasons surfaced by ``_strict_validation_rejection_reason``.

Core's hook names element-validator rejections on the config path only; a pattern-path
match_guard rejection and a missing required_when key both end in the generic
no-feature-groups error with nothing named. These tests pin the reasons the mixin adds,
and that non-candidates stay silent while reporting itself never raises.
"""

from __future__ import annotations

from mloda.core.abstract_plugins.components.options import Options

from mloda.community.feature_groups.data_operations.row_preserving.ffill.pyarrow_ffill import PyArrowFfill
from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.pandas_frame_aggregate import (
    PandasFrameAggregate,
)
from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.pyarrow_scalar_arithmetic import (
    PyArrowScalarArithmetic,
)


class TestPatternPathGuardRejectionReported:
    """A pattern-path match_guard rejection must be named, not a silent non-match."""

    def test_mistyped_constant_reports_a_reason(self) -> None:
        options = Options(context={"constant": "five"})
        reason = PyArrowScalarArithmetic._strict_validation_rejection_reason("value_int__add_constant", options)
        assert reason is not None
        assert "match_guard" in reason
        assert "'constant'" in reason
        assert "'five'" in reason

    def test_nested_singleton_constant_reports_a_reason(self) -> None:
        """A double-wrapped scalar is guard-rejected; the arity carve-out must not swallow it."""
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
        assert "match_guard" in reason
        assert "'order_by'" in reason

    def test_non_candidate_with_guard_rejected_value_reports_nothing(self) -> None:
        """No pattern name and no in_features: ffill was never a candidate, so nothing is reported."""
        options = Options(context={"order_by": 123})
        assert PyArrowFfill._strict_validation_rejection_reason("some_unrelated_feature", options) is None


class TestMissingRequiredWhenReported:
    """A missing required_when key must be named, not left as a debug log."""

    def test_missing_order_by_reports_a_reason(self) -> None:
        reason = PyArrowFfill._strict_validation_rejection_reason("value_float__ffill", Options())
        assert reason is not None
        assert "required option 'order_by'" in reason
        assert "propagate_context_keys" in reason

    def test_present_order_by_reports_nothing(self) -> None:
        options = Options(context={"order_by": "ts"})
        assert PyArrowFfill._strict_validation_rejection_reason("value_float__ffill", options) is None


class TestRejectionReasonHookNeverRaises:
    """The hook is diagnostics only: a hostile value repr must not escape as an exception."""

    def test_unreprable_guard_rejected_value_still_reports_a_reason(self) -> None:
        class ExplodingRepr:
            """Value whose repr raises, so reporting must survive the formatting failure."""

            def __repr__(self) -> str:
                raise RuntimeError("repr exploded")

        options = Options(context={"order_by": ExplodingRepr()})
        reason = PyArrowFfill._strict_validation_rejection_reason("value_float__ffill", options)
        assert reason is not None
        assert "'order_by'" in reason


class TestFrameAggregateNamePathReportsNothing:
    """frame_aggregate parses size and unit from the name, so its name path has nothing to report."""

    def test_rolling_name_skips_option_driven_required_when(self) -> None:
        """The name supplies size and unit, so a frame_type option cannot demand them: the match holds silently."""
        options = Options(context={"partition_by": ["region"], "order_by": "timestamp", "frame_type": "time"})
        assert PandasFrameAggregate.match_feature_group_criteria("sales__sum_rolling_3", options)
        assert PandasFrameAggregate._strict_validation_rejection_reason("sales__sum_rolling_3", options) is None
