"""Structural guard that both arithmetic families share one base class.

The point- and scalar-arithmetic feature groups both subclass a single base
(``ArithmeticFeatureGroupBase`` in ``data_operations/arithmetic_base.py``) that
holds the common arithmetic-op extraction and numeric-source skeleton.
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
    """Both arithmetic families derive from the shared ``ArithmeticFeatureGroupBase``."""

    def test_point_arithmetic_subclasses_base(self) -> None:
        """``PointArithmeticFeatureGroup`` inherits from the shared base."""
        assert issubclass(PointArithmeticFeatureGroup, ArithmeticFeatureGroupBase)

    def test_scalar_arithmetic_subclasses_base(self) -> None:
        """``ScalarArithmeticFeatureGroup`` inherits from the shared base."""
        assert issubclass(ScalarArithmeticFeatureGroup, ArithmeticFeatureGroupBase)
