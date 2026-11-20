"""Tests for PythonDictAggregation compute implementation."""

from __future__ import annotations

import math
from typing import Any

from mloda.user import DataType, Options

from mloda.community.feature_groups.data_operations.aggregation.base import AggregationFeatureGroup
from mloda.community.feature_groups.data_operations.aggregation.python_dict_aggregation import (
    PythonDictAggregation,
)
from mloda.testing.feature_groups.data_operations.aggregation.aggregation import (
    AggregationTestBase,
)
from mloda.testing.feature_groups.data_operations.helpers import arrow_table_of
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.frameworks import PythonDictTestMixin
from mloda.testing.typed_data import TypedTestData


class TestPythonDictAggregation(CapabilityHookTestMixin, PythonDictTestMixin, AggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictAggregation

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
    the correct target behavior. ``reference_aggregation`` is pure Python and
    groups NaN keys together, so it agrees with that and is a valid oracle here.
    """

    def test_nan_partition_keys_grouped_together_matches_pyarrow_oracle(self) -> None:
        from mloda.testing.feature_groups.data_operations.aggregation.reference import reference_aggregation
        from mloda.testing.feature_groups.data_operations.helpers import (
            calculate_with,
            extract_column,
            make_feature_set,
        )

        typed = TypedTestData(
            {
                "grp": (DataType.DOUBLE, [float("nan"), float("nan"), 1.0, 2.0]),
                "val": (DataType.DOUBLE, [10.0, 20.0, 100.0, 200.0]),
            }
        )
        data = typed.to_python_dict()
        fs = make_feature_set("val__sum_agg", ["grp"])

        result = calculate_with(PythonDictAggregation, data, fs)
        result_val_col = extract_column(result, "val__sum_agg")
        result_grp_col = extract_column(result, "grp")
        assert len(result_val_col) == 3, f"expected 3 groups (the two NaN rows merged into one), got {result_val_col!r}"
        nan_sums = [v for k, v in zip(result_grp_col, result_val_col) if isinstance(k, float) and math.isnan(k)]
        assert nan_sums == [30.0], (
            f"expected the two NaN-keyed rows (values 10.0 and 20.0) to merge into one group summing "
            f"to 30.0 (matching PyArrow's grouping), got {nan_sums!r}"
        )

        arrow_table = arrow_table_of(typed)
        oracle = reference_aggregation(AggregationFeatureGroup, arrow_table, fs)
        oracle_val_col = extract_column(oracle, "val__sum_agg")
        assert len(result_val_col) == len(oracle_val_col), (
            f"PythonDict={len(result_val_col)} groups {result_val_col!r} vs "
            f"PyArrow oracle={len(oracle_val_col)} groups {oracle_val_col!r}"
        )


class TestPythonDictMinMaxSkipsNan:
    """min/max must skip NaN values within a group, not propagate them.

    ``PythonDictAggregation._reduce`` (python_dict_aggregation.py lines
    134-137) reduces a group's non-null values with Python's builtin
    ``min()``/``max()``. Python's builtins short-circuit to NaN the moment
    any element is NaN (``min([nan, 1.0, 3.0]) == nan``), but PyArrow's
    ``pc.min``/``pc.max`` skip NaN and return the true minimum/maximum among
    the non-NaN values, as does the pure Python ``reference_aggregation``.
    """

    _TYPED = TypedTestData(
        {"grp": (DataType.STRING, ["A", "A", "A"]), "val": (DataType.DOUBLE, [float("nan"), 1.0, 3.0])}
    )

    @classmethod
    def _backend_val(cls, agg_type: str) -> Any:
        from mloda.testing.feature_groups.data_operations.helpers import (
            calculate_with,
            extract_column,
            make_feature_set,
        )

        fs = make_feature_set(f"val__{agg_type}_agg", ["grp"])
        result = calculate_with(PythonDictAggregation, cls._TYPED.to_python_dict(), fs)
        return extract_column(result, f"val__{agg_type}_agg")[0]

    @classmethod
    def _oracle_val(cls, agg_type: str) -> Any:
        from mloda.testing.feature_groups.data_operations.aggregation.reference import reference_aggregation
        from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set

        fs = make_feature_set(f"val__{agg_type}_agg", ["grp"])
        oracle = reference_aggregation(AggregationFeatureGroup, arrow_table_of(cls._TYPED), fs)
        return extract_column(oracle, f"val__{agg_type}_agg")[0]

    def test_min_skips_nan_matches_pyarrow_oracle(self) -> None:
        result_val = self._backend_val("min")
        assert result_val == 1.0, f"PythonDict min must skip NaN and be 1.0, got {result_val!r}"
        oracle_val = self._oracle_val("min")
        assert result_val == oracle_val, f"PythonDict min={result_val!r} != PyArrow oracle min={oracle_val!r}"

    def test_max_skips_nan_matches_pyarrow_oracle(self) -> None:
        result_val = self._backend_val("max")
        assert result_val == 3.0, f"PythonDict max must skip NaN and be 3.0, got {result_val!r}"
        oracle_val = self._oracle_val("max")
        assert result_val == oracle_val, f"PythonDict max={result_val!r} != PyArrow oracle max={oracle_val!r}"
