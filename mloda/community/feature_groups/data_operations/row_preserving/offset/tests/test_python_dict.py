"""Tests for PythonDictOffset compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.offset.python_dict_offset import (
    PythonDictOffset,
)
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.offset.offset import OffsetTestBase


class TestPythonDictOffset(PythonDictTestMixin, OffsetTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictOffset


class TestPythonDictOffsetNanPartitionKeyGrouping:
    """A NaN partition-key value must group with itself, not split into singletons.

    Two rows that both carry ``float('nan')`` in a partition column compare unequal
    (``nan != nan``), so building the group key from the raw partition value would split
    them into separate one-row groups instead of one shared group, and ``lag_1`` on a
    one-row group is always ``None``. ``offset`` has no PyArrow production backend;
    ``ReferenceOffset`` (the cross-framework test reference) builds its group key the exact
    same way and reproduces this identical bug, so it is NOT a valid oracle here. Instead,
    this test asks PyArrow's own ``Table.group_by()`` directly which rows it considers one
    partition, and derives the expected ``lag_1`` value from that by hand.
    """

    def test_nan_partition_rows_share_one_group_lag_carries_forward(self) -> None:
        import math

        import pyarrow as pa

        from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set

        arrow_table = pa.table(
            {
                "grp": pa.array([float("nan"), float("nan"), 1.0, 2.0], type=pa.float64()),
                "ord": pa.array([1, 2, 3, 4], type=pa.int64()),
                "val": pa.array([10.0, 20.0, 100.0, 200.0], type=pa.float64()),
            }
        )

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
        fs = make_feature_set("val__lag_1_offset", ["grp"], "ord")
        result = PythonDictOffset.calculate_feature(data, fs)
        result_col = extract_column(result, "val__lag_1_offset")

        # Row 1 (ord=2, val=20.0) is second (by ord) in the shared NaN partition PyArrow
        # reports above, so its lag_1 must carry row 0's value (10.0) instead of None.
        assert result_col[1] == 10.0, (
            f"expected row 1's lag_1 to carry the shared NaN partition's first value "
            f"(10.0, matching PyArrow's own group_by()), got {result_col[1]!r} (PythonDict "
            "split the NaN rows into separate one-row groups)"
        )
