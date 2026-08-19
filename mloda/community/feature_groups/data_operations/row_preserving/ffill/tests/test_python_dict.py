"""Tests for PythonDictFfill compute implementation."""

from __future__ import annotations

from typing import Any

from mloda.community.feature_groups.data_operations.row_preserving.ffill.python_dict_ffill import (
    PythonDictFfill,
)
from mloda.testing.feature_groups.data_operations.mixins.python_dict import PythonDictTestMixin
from mloda.testing.feature_groups.data_operations.row_preserving.ffill.ffill import (
    FfillTestBase,
)


class TestPythonDictFfill(PythonDictTestMixin, FfillTestBase):
    """All tests inherited from the base class."""

    @classmethod
    def implementation_class(cls) -> Any:
        return PythonDictFfill


class TestPythonDictFfillNanPartitionKeyGrouping:
    """A NaN partition-key value must group with itself, not split into singletons.

    Two rows that both carry ``float('nan')`` in a partition column compare unequal
    (``nan != nan``), so building the group key from the raw partition value would split
    them into separate one-row groups instead of one shared group, and forward-fill within
    a one-row group can never carry a value into it. This test asks PyArrow's own
    ``Table.group_by()`` directly which rows it considers one partition, and derives the
    expected forward-filled value from that by hand (see
    ffill/tests/test_pyarrow.py::TestPyArrowFfillNanPartitionKeyGrouping for the equivalent
    check against ``PyArrowFfill`` itself, which merges NaN partition keys too).
    """

    def test_nan_partition_rows_share_one_group_ffill_carries_forward(self) -> None:
        import math

        import pyarrow as pa

        from mloda.testing.feature_groups.data_operations.helpers import extract_column, make_feature_set

        arrow_table = pa.table(
            {
                "grp": pa.array([float("nan"), float("nan"), 1.0, 2.0], type=pa.float64()),
                "ord": pa.array([1, 2, 3, 4], type=pa.int64()),
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
        data["val"] = [10.0, None, 100.0, None]
        fs = make_feature_set("val__ffill", ["grp"], "ord")
        result = PythonDictFfill.calculate_feature(data, fs)
        result_col = extract_column(result, "val__ffill")

        # Row 1 (ord=2, val=None) is second (by ord) in the shared NaN partition PyArrow
        # reports above, so it must carry row 0's value (10.0) forward instead of staying None.
        assert result_col[1] == 10.0, (
            f"expected row 1's ffill to carry the shared NaN partition's value forward "
            f"(10.0, matching PyArrow's own group_by()), got {result_col[1]!r} (PythonDict "
            "split the NaN rows into separate one-row groups)"
        )
