"""Tests for PythonDictPointArithmetic compute implementation."""

from __future__ import annotations

import math
from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.python_dict_point_arithmetic import (
    PythonDictPointArithmetic,
)
from mloda.testing.feature_groups.data_operations.row_preserving.point_arithmetic.point_arithmetic import (
    PointArithmeticTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin


class TestPythonDictPointArithmetic(PythonDictTestMixin, PointArithmeticTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictPointArithmetic


class TestPythonDictDivideSignedZeroAndNan:
    """Divide must get IEEE-754 sign/NaN right for negative-zero denominators.

    ``python_dict_point_arithmetic._div`` treats ``b == 0.0`` as true for BOTH
    ``+0.0`` and ``-0.0``, and always derives the result sign from the
    numerator alone via ``math.copysign(inf, a)`` without checking whether
    ``a`` is NaN first. That is wrong on two counts:

    - the sign of an inf result must come from ``sign(a) * sign(b)``, not
      ``sign(a)`` alone, so ``1.0 / -0.0`` must be ``-inf`` (current code
      gives ``+inf``) and ``-1.0 / -0.0`` must be ``+inf`` (current code
      happens to also give ``+inf``, but for the wrong reason);
    - a NaN numerator must produce NaN regardless of the denominator's sign,
      so ``nan / 0.0`` must be ``nan`` (current code gives ``+inf`` because
      ``math.copysign(inf, nan)`` is ``inf``, not ``nan``).

    Each test compares against the live PyArrow reference
    (``pyarrow.compute.divide``, the four-backend IEEE-754 majority
    behaviour) rather than a hand-picked literal.
    """

    @staticmethod
    def _divide_via(a: float, b: float) -> tuple[Any, Any]:
        """Run PythonDict and the live PyArrow oracle for one (a, b) divide row; return (result, oracle)."""
        import pyarrow as pa

        from mloda.community.feature_groups.data_operations.row_preserving.point_arithmetic.pyarrow_point_arithmetic import (
            PyArrowPointArithmetic,
        )
        from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set

        arrow_table = pa.table(
            {
                "a": pa.array([a], type=pa.float64()),
                "b": pa.array([b], type=pa.float64()),
            }
        )
        data = {name: arrow_table.column(name).to_pylist() for name in arrow_table.column_names}
        fs = make_feature_set("a&b__divide_point")

        result = PythonDictPointArithmetic.calculate_feature(data, fs)
        oracle = PyArrowPointArithmetic.calculate_feature(arrow_table, fs)

        result_val = extract_column(result, "a&b__divide_point")[0]
        oracle_val = extract_column(oracle, "a&b__divide_point")[0]
        return result_val, oracle_val

    def test_positive_divided_by_negative_zero_matches_pyarrow_oracle(self) -> None:
        """``1.0 / -0.0``: PyArrow gives ``-inf``; the buggy helper gives ``+inf``."""
        result_val, oracle_val = self._divide_via(1.0, -0.0)

        assert math.isinf(oracle_val) and math.copysign(1.0, oracle_val) < 0, (
            f"expected PyArrow oracle to be -inf, got {oracle_val!r}"
        )
        assert math.isinf(result_val) and math.copysign(1.0, result_val) == math.copysign(1.0, oracle_val), (
            f"PythonDict divide sign mismatch: got {result_val!r}, PyArrow oracle {oracle_val!r}"
        )

    def test_negative_divided_by_negative_zero_matches_pyarrow_oracle(self) -> None:
        """``-1.0 / -0.0``: PyArrow gives ``+inf``."""
        result_val, oracle_val = self._divide_via(-1.0, -0.0)

        assert math.isinf(oracle_val) and math.copysign(1.0, oracle_val) > 0, (
            f"expected PyArrow oracle to be +inf, got {oracle_val!r}"
        )
        assert math.isinf(result_val) and math.copysign(1.0, result_val) == math.copysign(1.0, oracle_val), (
            f"PythonDict divide sign mismatch: got {result_val!r}, PyArrow oracle {oracle_val!r}"
        )

    def test_nan_divided_by_zero_matches_pyarrow_oracle(self) -> None:
        """``nan / 0.0``: PyArrow gives ``nan``; the buggy helper gives ``+inf``."""
        result_val, oracle_val = self._divide_via(float("nan"), 0.0)

        assert oracle_val is not None and math.isnan(oracle_val), (
            f"expected PyArrow oracle to be nan, got {oracle_val!r}"
        )
        assert result_val is not None and math.isnan(result_val), (
            f"PythonDict divide expected nan (matching PyArrow oracle), got {result_val!r}"
        )
