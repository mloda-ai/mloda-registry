"""Tests for PythonDict window aggregation compute implementation."""

from __future__ import annotations

import math
from typing import Any

from mloda.user import Options

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.python_dict_window_aggregation import (
    PythonDictWindowAggregation,
)
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation.window_aggregation import (
    WindowAggregationTestBase,
)


class TestPythonDictWindowAggregation(CapabilityHookTestMixin, PythonDictTestMixin, WindowAggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictWindowAggregation

    @classmethod
    def supported_agg_types(cls) -> set[str]:
        return {*cls.ALL_AGG_TYPES, "mean"}

    @classmethod
    def capability_supported(cls) -> tuple[tuple[str, Options], ...]:
        return (
            ("value__median_window", Options()),
            ("value__mode_window", Options()),
        )


class TestPythonDictNanPartitionKeyGrouping:
    """A NaN partition-key value must group with itself, not split into singletons.

    ``PythonDictWindowAggregation._compute_window`` builds the group key as
    ``tuple(col[i] for col in partition_cols)`` (python_dict_window_aggregation.py
    line 107), the row-preserving twin of the same construct in
    ``python_dict_aggregation.py``. Two rows that both carry ``float('nan')``
    in a partition column compare unequal (``nan != nan``), so the dict
    splits them into separate one-row groups instead of one shared group,
    and each row is broadcast only its own (unreduced) value.

    Verified empirically: PyArrow's own ``Table.group_by()`` merges all NaN
    keys of a column into a single group, so "one shared NaN group" is the
    correct target. ``ReferenceWindowAggregation`` drives its grouping
    through PyArrow's real ``group_by().aggregate()``, so it is a valid
    oracle here.
    """

    def test_nan_partition_keys_grouped_together_matches_pyarrow_oracle(self) -> None:
        import pyarrow as pa

        from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set
        from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation.reference import (
            ReferenceWindowAggregation,
        )

        arrow_table = pa.table(
            {
                "grp": pa.array([float("nan"), float("nan"), 1.0, 2.0], type=pa.float64()),
                "val": pa.array([10.0, 20.0, 100.0, 200.0], type=pa.float64()),
            }
        )
        data = {name: arrow_table.column(name).to_pylist() for name in arrow_table.column_names}
        fs = make_feature_set("val__sum_window", ["grp"])

        result = PythonDictWindowAggregation.calculate_feature(data, fs)
        oracle = ReferenceWindowAggregation.calculate_feature(arrow_table, fs)

        result_col = extract_column(result, "val__sum_window")
        oracle_col = extract_column(oracle, "val__sum_window")

        assert oracle_col == [30.0, 30.0, 100.0, 200.0], f"sanity check on the oracle itself failed: {oracle_col!r}"
        grp_col = extract_column(result, "grp")
        nan_broadcast = [v for k, v in zip(grp_col, result_col) if isinstance(k, float) and math.isnan(k)]
        assert nan_broadcast == [30.0, 30.0], (
            f"expected both NaN-keyed rows to broadcast the shared group's sum (10.0 + 20.0 = 30.0, "
            f"matching the PyArrow oracle {oracle_col!r}), got {nan_broadcast!r}"
        )


class TestPythonDictMinMaxSkipsNan:
    """min/max must skip NaN values within a group, not propagate them.

    ``PythonDictWindowAggregation._reduce`` (python_dict_window_aggregation.py
    lines 152-155) reduces a group's non-null values with Python's builtin
    ``min()``/``max()``, which short-circuits to NaN the moment any element
    is NaN. PyArrow's ``pc.min``/``pc.max`` (exercised here through
    ``ReferenceWindowAggregation``'s ``group_by().aggregate()`` call) skip
    NaN and return the true minimum/maximum among the non-NaN values.
    """

    @staticmethod
    def _agg_via(agg_type: str) -> tuple[Any, Any]:
        import pyarrow as pa

        from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set
        from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation.reference import (
            ReferenceWindowAggregation,
        )

        arrow_table = pa.table(
            {
                "grp": pa.array(["A", "A", "A"], type=pa.string()),
                "val": pa.array([float("nan"), 1.0, 3.0], type=pa.float64()),
            }
        )
        data = {name: arrow_table.column(name).to_pylist() for name in arrow_table.column_names}
        fs = make_feature_set(f"val__{agg_type}_window", ["grp"])

        result = PythonDictWindowAggregation.calculate_feature(data, fs)
        oracle = ReferenceWindowAggregation.calculate_feature(arrow_table, fs)

        result_val = extract_column(result, f"val__{agg_type}_window")[0]
        oracle_val = extract_column(oracle, f"val__{agg_type}_window")[0]
        return result_val, oracle_val

    def test_min_skips_nan_matches_pyarrow_oracle(self) -> None:
        result_val, oracle_val = self._agg_via("min")
        assert oracle_val == 1.0, f"expected PyArrow oracle min to skip NaN and be 1.0, got {oracle_val!r}"
        assert result_val == oracle_val, f"PythonDict min={result_val!r} != PyArrow oracle min={oracle_val!r}"

    def test_max_skips_nan_matches_pyarrow_oracle(self) -> None:
        result_val, oracle_val = self._agg_via("max")
        assert oracle_val == 3.0, f"expected PyArrow oracle max to skip NaN and be 3.0, got {oracle_val!r}"
        assert result_val == oracle_val, f"PythonDict max={result_val!r} != PyArrow oracle max={oracle_val!r}"
