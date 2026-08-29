"""Tests for PythonDictScalarAggregate compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.user import Options
from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.python_dict_scalar_aggregate import (
    PythonDictScalarAggregate,
)
from mloda.testing.feature_groups.data_operations.row_preserving.scalar_aggregate.scalar_aggregate import (
    ScalarAggregateTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin


class TestPythonDictScalarAggregate(CapabilityHookTestMixin, PythonDictTestMixin, ScalarAggregateTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictScalarAggregate

    @classmethod
    def capability_supported(cls) -> tuple[tuple[str, Options], ...]:
        return (("value__median_scalar", Options()),)


class TestPythonDictMinMaxSkipsNan:
    """min/max must skip NaN values in the column, not propagate them.

    ``PythonDictScalarAggregate._reduce`` (python_dict_scalar_aggregate.py
    lines 87-90) reduces the whole (already null-filtered) column with
    Python's builtin ``min()``/``max()``, which short-circuits to NaN the
    moment any element is NaN (``min([nan, 1.0, 3.0]) == nan``). PyArrow's
    ``pc.min``/``pc.max`` (used by the production ``PyArrowScalarAggregate``
    backend) skip NaN and return the true minimum/maximum among the non-NaN
    values.
    """

    @staticmethod
    def _agg_via(agg_type: str) -> tuple[Any, Any]:
        import pyarrow as pa

        from mloda.community.feature_groups.data_operations.row_preserving.scalar_aggregate.pyarrow_scalar_aggregate import (
            PyArrowScalarAggregate,
        )
        from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set

        arrow_table = pa.table({"val": pa.array([float("nan"), 1.0, 3.0], type=pa.float64())})
        data = {name: arrow_table.column(name).to_pylist() for name in arrow_table.column_names}
        fs = make_feature_set(f"val__{agg_type}_scalar")

        result = PythonDictScalarAggregate.calculate_feature(data, fs)
        oracle = PyArrowScalarAggregate.calculate_feature(arrow_table, fs)

        result_val = extract_column(result, f"val__{agg_type}_scalar")[0]
        oracle_val = extract_column(oracle, f"val__{agg_type}_scalar")[0]
        return result_val, oracle_val

    def test_min_skips_nan_matches_pyarrow_oracle(self) -> None:
        result_val, oracle_val = self._agg_via("min")
        assert oracle_val == 1.0, f"expected PyArrow's pc.min to skip NaN and be 1.0, got {oracle_val!r}"
        assert result_val == oracle_val, f"PythonDict min={result_val!r} != PyArrow oracle min={oracle_val!r}"

    def test_max_skips_nan_matches_pyarrow_oracle(self) -> None:
        result_val, oracle_val = self._agg_via("max")
        assert oracle_val == 3.0, f"expected PyArrow's pc.max to skip NaN and be 3.0, got {oracle_val!r}"
        assert result_val == oracle_val, f"PythonDict max={result_val!r} != PyArrow oracle max={oracle_val!r}"
