"""Tests for PythonDict window aggregation compute implementation."""

from __future__ import annotations

import math
from typing import Any

from mloda.user import DataType, Options

from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.base import (
    WindowAggregationFeatureGroup,
)
from mloda.community.feature_groups.data_operations.row_preserving.window_aggregation.python_dict_window_aggregation import (
    PythonDictWindowAggregation,
)
from mloda.testing.feature_groups.data_operations.helpers import arrow_table_of
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.frameworks import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation.window_aggregation import (
    WindowAggregationTestBase,
)
from mloda.testing.typed_data import TypedTestData


class TestPythonDictWindowAggregation(CapabilityHookTestMixin, PythonDictTestMixin, WindowAggregationTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictWindowAggregation

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
    correct target. ``reference_window_aggregation`` is pure Python and
    groups NaN keys together, so it is a valid oracle here.
    """

    def test_nan_partition_keys_grouped_together_matches_pyarrow_oracle(self) -> None:

        from mloda.testing.feature_groups.data_operations.helpers import (
            calculate_with,
            extract_column,
            make_feature_set,
        )
        from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation.reference import (
            reference_window_aggregation,
        )

        typed = TypedTestData(
            {
                "grp": (DataType.DOUBLE, [float("nan"), float("nan"), 1.0, 2.0]),
                "val": (DataType.DOUBLE, [10.0, 20.0, 100.0, 200.0]),
            }
        )
        data = typed.to_python_dict()
        fs = make_feature_set("val__sum_window", ["grp"])

        result = calculate_with(PythonDictWindowAggregation, data, fs)
        result_col = extract_column(result, "val__sum_window")
        grp_col = extract_column(result, "grp")
        nan_broadcast = [v for k, v in zip(grp_col, result_col) if isinstance(k, float) and math.isnan(k)]
        assert nan_broadcast == [30.0, 30.0], (
            f"expected both NaN-keyed rows to broadcast the shared group's sum (10.0 + 20.0 = 30.0, "
            f"matching the PyArrow oracle), got {nan_broadcast!r}"
        )

        oracle = reference_window_aggregation(WindowAggregationFeatureGroup, arrow_table_of(typed), fs)
        oracle_col = extract_column(oracle, "val__sum_window")
        assert oracle_col == [30.0, 30.0, 100.0, 200.0], f"sanity check on the oracle itself failed: {oracle_col!r}"


class TestPythonDictMinMaxSkipsNan:
    """min/max must skip NaN values within a group, not propagate them.

    ``PythonDictWindowAggregation._reduce`` (python_dict_window_aggregation.py
    lines 152-155) reduces a group's non-null values with Python's builtin
    ``min()``/``max()``, which short-circuits to NaN the moment any element
    is NaN. PyArrow's ``pc.min``/``pc.max`` skip NaN and return the true minimum/maximum
    among the non-NaN values, as does the pure Python ``reference_window_aggregation``.
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

        fs = make_feature_set(f"val__{agg_type}_window", ["grp"])
        result = calculate_with(PythonDictWindowAggregation, cls._TYPED.to_python_dict(), fs)
        return extract_column(result, f"val__{agg_type}_window")[0]

    @classmethod
    def _oracle_val(cls, agg_type: str) -> Any:
        from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set
        from mloda.testing.feature_groups.data_operations.row_preserving.window_aggregation.reference import (
            reference_window_aggregation,
        )

        fs = make_feature_set(f"val__{agg_type}_window", ["grp"])
        oracle = reference_window_aggregation(WindowAggregationFeatureGroup, arrow_table_of(cls._TYPED), fs)
        return extract_column(oracle, f"val__{agg_type}_window")[0]

    def test_min_skips_nan_matches_pyarrow_oracle(self) -> None:
        result_val = self._backend_val("min")
        assert result_val == 1.0, f"PythonDict min must skip NaN and be 1.0, got {result_val!r}"
        oracle_val = self._oracle_val("min")
        assert oracle_val == 1.0, f"expected PyArrow oracle min to skip NaN and be 1.0, got {oracle_val!r}"
        assert result_val == oracle_val, f"PythonDict min={result_val!r} != PyArrow oracle min={oracle_val!r}"

    def test_max_skips_nan_matches_pyarrow_oracle(self) -> None:
        result_val = self._backend_val("max")
        assert result_val == 3.0, f"PythonDict max must skip NaN and be 3.0, got {result_val!r}"
        oracle_val = self._oracle_val("max")
        assert oracle_val == 3.0, f"expected PyArrow oracle max to skip NaN and be 3.0, got {oracle_val!r}"
        assert result_val == oracle_val, f"PythonDict max={result_val!r} != PyArrow oracle max={oracle_val!r}"
