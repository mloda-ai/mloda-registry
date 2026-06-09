"""Structural guard that both arithmetic families share one base class.

The point- and scalar-arithmetic feature groups both subclass a single base
(``ArithmeticFeatureGroupBase`` in ``data_operations/arithmetic/base.py``) that
holds the common arithmetic-op extraction and numeric-source skeleton.
"""

from __future__ import annotations

from mloda.community.feature_groups.data_operations.arithmetic.base import (
    ARITHMETIC_OP_NAMES,
    ArithmeticFeatureGroupBase,
)
from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.base import (
    ARITHMETIC_OPERATIONS as POINT_ARITHMETIC_OPERATIONS,
)
from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.base import (
    PointArithmeticFeatureGroup,
)
from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
    ARITHMETIC_OPERATIONS as SCALAR_ARITHMETIC_OPERATIONS,
)
from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
    ScalarArithmeticFeatureGroup,
)


class TestArithmeticFeatureGroupBase:
    """Both arithmetic families derive from the shared ``ArithmeticFeatureGroupBase``."""

    def test_point_arithmetic_subclasses_base(self) -> None:
        """``PointArithmeticFeatureGroup`` inherits from the shared base."""
        assert issubclass(PointArithmeticFeatureGroup, ArithmeticFeatureGroupBase)

    def test_scalar_arithmetic_subclasses_base(self) -> None:
        """``ScalarArithmeticFeatureGroup`` inherits from the shared base."""
        assert issubclass(ScalarArithmeticFeatureGroup, ArithmeticFeatureGroupBase)


class TestArithmeticOpNamesSingleSourceOfTruth:
    """Guard that each family's op set stays in lockstep with the shared ``ARITHMETIC_OP_NAMES``.

    Each family keeps its own ``ARITHMETIC_OPERATIONS`` dict whose KEYS are the
    op names (mapped to family-specific descriptions), but the matcher accepts
    an op only if it is in the module-level ``ARITHMETIC_OP_NAMES`` frozenset in
    ``arithmetic.base`` (used by ``_validate_string_match``).
    If a family added an op to its dict without it being in the shared frozenset
    (or vice versa), the op would be silently rejected at match time: exactly
    the divergence class issue #214 targets. These structural guards pin the
    key sets to the single source of truth and pin each family's non-empty
    ``OPERATION_LABEL`` (used in the numeric-source rejection message).
    """

    def test_point_arithmetic_operation_keys_match_shared_op_names(self) -> None:
        """The point family's ``ARITHMETIC_OPERATIONS`` keys equal the shared op names."""
        assert set(POINT_ARITHMETIC_OPERATIONS) == ARITHMETIC_OP_NAMES

    def test_scalar_arithmetic_operation_keys_match_shared_op_names(self) -> None:
        """The scalar family's ``ARITHMETIC_OPERATIONS`` keys equal the shared op names."""
        assert set(SCALAR_ARITHMETIC_OPERATIONS) == ARITHMETIC_OP_NAMES

    def test_point_arithmetic_operation_label(self) -> None:
        """``PointArithmeticFeatureGroup`` sets a non-empty, family-specific ``OPERATION_LABEL``."""
        assert PointArithmeticFeatureGroup.OPERATION_LABEL == "point arithmetic"
        assert PointArithmeticFeatureGroup.OPERATION_LABEL

    def test_scalar_arithmetic_operation_label(self) -> None:
        """``ScalarArithmeticFeatureGroup`` sets a non-empty, family-specific ``OPERATION_LABEL``."""
        assert ScalarArithmeticFeatureGroup.OPERATION_LABEL == "scalar arithmetic"
        assert ScalarArithmeticFeatureGroup.OPERATION_LABEL
