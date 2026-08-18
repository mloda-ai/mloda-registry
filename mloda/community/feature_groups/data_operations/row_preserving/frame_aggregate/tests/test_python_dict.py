"""Tests for PythonDict frame aggregate implementation.

Uses the unified FrameAggregateTestBase.
"""

from __future__ import annotations

import math
from typing import Any

from mloda.core.abstract_plugins.components.options import Options
from mloda.testing.feature_groups.data_operations.mixins.capability import CapabilityHookTestMixin
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.frame_aggregate.frame_aggregate import (
    FrameAggregateTestBase,
    time_frame_options,
)

from mloda.community.feature_groups.data_operations.row_preserving.frame_aggregate.python_dict_frame_aggregate import (
    PythonDictFrameAggregate,
)


class TestPythonDictFrameAggregate(CapabilityHookTestMixin, PythonDictTestMixin, FrameAggregateTestBase):
    """Unified tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictFrameAggregate

    @classmethod
    def capability_supported(cls) -> tuple[tuple[str, Options], ...]:
        return (
            ("value_time_frame", time_frame_options("month")),
            ("value__median_rolling_3", Options()),
        )


class TestPythonDictNanPartitionKeyGrouping:
    """A NaN partition-key value must group with itself, not split into singletons.

    ``PythonDictFrameAggregate._compute_frame`` builds the group key as
    ``tuple(col[i] for col in partition_cols)`` (python_dict_frame_aggregate.py
    line 126). Two rows that both carry ``float('nan')`` in a partition
    column compare unequal (``nan != nan``), so the dict splits them into
    separate one-row groups/windows instead of one shared partition.

    Note: this operation's own cross-framework test reference,
    ``ReferenceFrameAggregate`` (row_preserving/frame_aggregate/reference.py),
    builds its group key the exact same way (``key = tuple(partition_lists[col][i]
    for col in partition_by)``) and therefore reproduces this identical bug --
    it is NOT a valid oracle here. Instead, this test asks PyArrow's own
    ``Table.group_by()`` directly which rows it considers one partition (the
    live, ground-truth grouping fact), and derives the expected windowed sum
    from that.
    """

    def test_nan_partition_rows_share_one_group_matches_pyarrow_group_by(self) -> None:
        import pyarrow as pa

        from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set

        arrow_table = pa.table(
            {
                "grp": pa.array([float("nan"), float("nan"), 1.0, 2.0], type=pa.float64()),
                "ts": pa.array([1, 2, 3, 4], type=pa.int64()),
                "val": pa.array([10.0, 20.0, 100.0, 200.0], type=pa.float64()),
            }
        )

        # Live PyArrow grouping fact: which rows does PyArrow's own group_by()
        # consider one partition for a NaN key?
        t_with_idx = arrow_table.append_column("__idx__", pa.array(range(arrow_table.num_rows)))
        grouped = t_with_idx.group_by(["grp"]).aggregate([("__idx__", "list")])
        idx_lists = grouped.column("__idx___list").to_pylist()
        keys = grouped.column("grp").to_pylist()
        nan_group_rows = next(rows for key, rows in zip(keys, idx_lists) if key is not None and math.isnan(key))
        assert nan_group_rows == [0, 1], (
            f"expected PyArrow's live group_by() to place both NaN-keyed rows (0, 1) in one "
            f"group, got {nan_group_rows!r}"
        )

        data = {name: arrow_table.column(name).to_pylist() for name in arrow_table.column_names}
        fs = make_feature_set("val__expanding_sum", ["grp"], order_by="ts")
        result = PythonDictFrameAggregate.calculate_feature(data, fs)
        result_col = extract_column(result, "val__expanding_sum")

        # Row 1 (ts=2, val=20.0) is second (by ts) in the shared NaN partition PyArrow
        # reports above, so its expanding sum must include row 0's value too: 10.0 + 20.0 = 30.0.
        assert result_col[1] == 30.0, (
            f"expected row 1's expanding sum to include both NaN-partition rows (10.0 + 20.0 = "
            f"30.0, matching PyArrow's own group_by()), got {result_col[1]!r} (PythonDict split "
            "the NaN rows into separate one-row groups)"
        )


class TestPythonDictMinMaxSkipsNan:
    """min/max must skip NaN values within a window, not propagate them.

    ``PythonDictFrameAggregate._reduce_window`` (python_dict_frame_aggregate.py
    lines 192-195) reduces a window's non-null values with Python's builtin
    ``min()``/``max()``, which short-circuits to NaN the moment any element
    is NaN. This operation's own ``ReferenceFrameAggregate`` test reference
    has the identical bug (it reduces windows via
    ``aggregation_helpers.aggregate()``, which also calls builtin
    ``min()``/``max()``), so it is NOT a valid oracle here either. This test
    instead calls PyArrow's own ``pyarrow.compute.min``/``max`` directly (the
    functions every genuine PyArrow-backed aggregation in this codebase
    delegates to) on the window's non-NaN values as the live oracle.
    """

    @staticmethod
    def _expanding_via(agg_type: str) -> tuple[list[Any], Any]:
        import pyarrow as pa
        import pyarrow.compute as pc

        from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set

        arrow_table = pa.table(
            {
                "grp": pa.array(["A", "A", "A"], type=pa.string()),
                "ts": pa.array([1, 2, 3], type=pa.int64()),
                "val": pa.array([float("nan"), 1.0, 3.0], type=pa.float64()),
            }
        )
        data = {name: arrow_table.column(name).to_pylist() for name in arrow_table.column_names}
        fs = make_feature_set(f"val__expanding_{agg_type}", ["grp"], order_by="ts")
        result = PythonDictFrameAggregate.calculate_feature(data, fs)
        result_col = extract_column(result, f"val__expanding_{agg_type}")

        pa_func = pc.min if agg_type == "min" else pc.max
        oracle_last = pa_func(pa.array([1.0, 3.0], type=pa.float64())).as_py()
        return result_col, oracle_last

    def test_expanding_min_skips_nan_matches_pyarrow_oracle(self) -> None:
        result_col, oracle_last = self._expanding_via("min")
        assert oracle_last == 1.0, f"expected PyArrow's pc.min to skip NaN and be 1.0, got {oracle_last!r}"
        assert result_col[-1] == oracle_last, (
            f"PythonDict expanding min at the last row = {result_col[-1]!r}, expected {oracle_last!r}"
        )

    def test_expanding_max_skips_nan_matches_pyarrow_oracle(self) -> None:
        result_col, oracle_last = self._expanding_via("max")
        assert oracle_last == 3.0, f"expected PyArrow's pc.max to skip NaN and be 3.0, got {oracle_last!r}"
        assert result_col[-1] == oracle_last, (
            f"PythonDict expanding max at the last row = {result_col[-1]!r}, expected {oracle_last!r}"
        )
