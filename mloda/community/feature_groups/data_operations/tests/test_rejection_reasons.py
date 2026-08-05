"""Rejection reasons surfaced by ``_strict_validation_rejection_reason``.

Core's hook names element-validator rejections on the config path only; a pattern-path
match_guard rejection and a missing required_when key both end in the generic
no-feature-groups error with nothing named. These tests pin the reasons the mixin adds.
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

    def test_valid_constant_reports_nothing(self) -> None:
        options = Options(context={"constant": 5})
        assert PyArrowScalarArithmetic._strict_validation_rejection_reason("value_int__add_constant", options) is None

    def test_unrelated_candidate_reports_nothing(self) -> None:
        assert PyArrowScalarArithmetic._strict_validation_rejection_reason("some_unrelated_feature", Options()) is None


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

    def test_config_path_guard_rejection_reports_a_reason(self) -> None:
        options = Options(context={"in_features": "value_float", "order_by": 123})
        reason = PyArrowFfill._strict_validation_rejection_reason("my_result", options)
        assert reason is not None
        assert "match_guard" in reason
        assert "'order_by'" in reason


class TestFrameAggregateNamePathReportsNothing:
    """frame_aggregate parses size and unit from the name, so its name path has nothing to report."""

    def test_rolling_name_with_required_context_reports_nothing(self) -> None:
        options = Options(context={"partition_by": ["region"], "order_by": "timestamp"})
        assert PandasFrameAggregate._strict_validation_rejection_reason("sales__sum_rolling_3", options) is None
