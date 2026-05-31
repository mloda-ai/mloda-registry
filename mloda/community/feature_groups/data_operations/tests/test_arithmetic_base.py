"""Tests for the shared arithmetic feature-group skeleton.

These tests encode the Definition of Done for issue #214: the point- and
scalar-arithmetic feature groups share a single base class
(``ArithmeticFeatureGroupBase`` in ``data_operations/arithmetic_base.py``)
that holds the common arithmetic-op extraction and numeric-source skeleton.
"""

from __future__ import annotations

from mloda.community.feature_groups.data_operations.arithmetic_base import ArithmeticFeatureGroupBase
from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.base import (
    PointArithmeticFeatureGroup,
)
from mloda.community.feature_groups.data_operations.row_preserving.scalar_arithmetic.base import (
    ScalarArithmeticFeatureGroup,
)


class TestArithmeticFeatureGroupBase:
    def test_arithmetic_op_constant(self) -> None:
        """The shared op key is exposed on the base class."""
        assert ArithmeticFeatureGroupBase.ARITHMETIC_OP == "arithmetic_op"

    def test_point_arithmetic_subclasses_base(self) -> None:
        """``PointArithmeticFeatureGroup`` inherits from the shared base."""
        assert issubclass(PointArithmeticFeatureGroup, ArithmeticFeatureGroupBase)

    def test_scalar_arithmetic_subclasses_base(self) -> None:
        """``ScalarArithmeticFeatureGroup`` inherits from the shared base."""
        assert issubclass(ScalarArithmeticFeatureGroup, ArithmeticFeatureGroupBase)
