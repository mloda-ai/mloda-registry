"""Tests for PythonDictAggregation compute implementation."""

from __future__ import annotations

import math
from typing import Any

from mloda.core.abstract_plugins.components.options import Options
from mloda.community.feature_groups.data_operations.aggregation.python_dict_aggregation import (
    PythonDictAggregation,
)
from mloda.testing.feature_groups.data_operations.aggregation.aggregation import (
    AggregationTestBase,
)
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin


class TestPythonDictAggregation(CapabilityHookTestMixin, PythonDictTestMixin, AggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictAggregation

    @classmethod
    def supported_agg_types(cls) -> set[str]:
        return {*cls.ALL_AGG_TYPES, "mean"}

    @classmethod
    def capability_supported(cls) -> tuple[tuple[str, Options], ...]:
        return (
            ("value__median_agg", Options()),
            ("value__mode_agg", Options()),
        )


class TestPythonDictNanPartitionKeyGrouping:
    """A NaN partition-key value must group with itself, not split into singletons.

    ``PythonDictAggregation._compute_group`` builds the group key as
    ``tuple(col[i] for col in partition_cols)`` (python_dict_aggregation.py
    line 94): a raw tuple of column values used directly as a dict key. Two
    rows that both carry ``float('nan')`` in a partition column compare
    unequal (``nan != nan``), so unless they are literally the same NaN
    object, the dict treats them as two distinct keys and splits them into
    separate one-row groups instead of one shared group.

    Verified empirically: PyArrow's own ``Table.group_by()`` merges all NaN
    keys of a column into a single group (distinct from a null/None group),
    so "one shared NaN group" -- not "one singleton group per NaN row" -- is
    the correct target behavior. ``ReferenceAggregation`` drives its grouping
    through PyArrow's real ``group_by().aggregate()``, so it reproduces that
    same live-PyArrow behavior and is a valid oracle here.
    """

    def test_nan_partition_keys_grouped_together_matches_pyarrow_oracle(self) -> None:
        import pyarrow as pa

        from mloda.testing.feature_groups.data_operations.aggregation.reference import (
            ReferenceAggregation,
        )
        from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set

        arrow_table = pa.table(
            {
                "grp": pa.array([float("nan"), float("nan"), 1.0, 2.0], type=pa.float64()),
                "val": pa.array([10.0, 20.0, 100.0, 200.0], type=pa.float64()),
            }
        )
        data = {name: arrow_table.column(name).to_pylist() for name in arrow_table.column_names}
        fs = make_feature_set("val__sum_agg", ["grp"])

        result = PythonDictAggregation.calculate_feature(data, fs)
        oracle = ReferenceAggregation.calculate_feature(arrow_table, fs)

        result_val_col = extract_column(result, "val__sum_agg")
        oracle_val_col = extract_column(oracle, "val__sum_agg")

        assert len(oracle_val_col) == 3, f"sanity check on the oracle itself failed: {oracle_val_col!r}"
        assert len(result_val_col) == len(oracle_val_col), (
            f"expected 3 groups (the two NaN rows merged into one, matching the PyArrow oracle), "
            f"got PythonDict={len(result_val_col)} groups {result_val_col!r} vs "
            f"PyArrow oracle={len(oracle_val_col)} groups {oracle_val_col!r}"
        )

        result_grp_col = extract_column(result, "grp")
        nan_sums = [v for k, v in zip(result_grp_col, result_val_col) if isinstance(k, float) and math.isnan(k)]
        assert nan_sums == [30.0], (
            f"expected the two NaN-keyed rows (values 10.0 and 20.0) to merge into one group summing "
            f"to 30.0 (matching PyArrow's grouping), got {nan_sums!r}"
        )


class TestPythonDictMinMaxSkipsNan:
    """min/max must skip NaN values within a group, not propagate them.

    ``PythonDictAggregation._reduce`` (python_dict_aggregation.py lines
    134-137) reduces a group's non-null values with Python's builtin
    ``min()``/``max()``. Python's builtins short-circuit to NaN the moment
    any element is NaN (``min([nan, 1.0, 3.0]) == nan``), but PyArrow's
    ``pc.min``/``pc.max`` (exercised here through ``ReferenceAggregation``'s
    ``group_by().aggregate()`` call) skip NaN and return the true minimum/
    maximum among the non-NaN values.
    """

    @staticmethod
    def _agg_via(agg_type: str) -> tuple[Any, Any]:
        import pyarrow as pa

        from mloda.testing.feature_groups.data_operations.aggregation.reference import (
            ReferenceAggregation,
        )
        from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set

        arrow_table = pa.table(
            {
                "grp": pa.array(["A", "A", "A"], type=pa.string()),
                "val": pa.array([float("nan"), 1.0, 3.0], type=pa.float64()),
            }
        )
        data = {name: arrow_table.column(name).to_pylist() for name in arrow_table.column_names}
        fs = make_feature_set(f"val__{agg_type}_agg", ["grp"])

        result = PythonDictAggregation.calculate_feature(data, fs)
        oracle = ReferenceAggregation.calculate_feature(arrow_table, fs)

        result_val = extract_column(result, f"val__{agg_type}_agg")[0]
        oracle_val = extract_column(oracle, f"val__{agg_type}_agg")[0]
        return result_val, oracle_val

    def test_min_skips_nan_matches_pyarrow_oracle(self) -> None:
        result_val, oracle_val = self._agg_via("min")
        assert oracle_val == 1.0, f"expected PyArrow oracle min to skip NaN and be 1.0, got {oracle_val!r}"
        assert result_val == oracle_val, f"PythonDict min={result_val!r} != PyArrow oracle min={oracle_val!r}"

    def test_max_skips_nan_matches_pyarrow_oracle(self) -> None:
        result_val, oracle_val = self._agg_via("max")
        assert oracle_val == 3.0, f"expected PyArrow oracle max to skip NaN and be 3.0, got {oracle_val!r}"
        assert result_val == oracle_val, f"PythonDict max={result_val!r} != PyArrow oracle max={oracle_val!r}"
